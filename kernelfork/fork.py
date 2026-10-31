"""In-place remask and two-sided fill for failed proof spans."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from kernelfork.lean_client import LeanClient
from kernelfork.spans import CharTokenizer
from kernelfork.types import ForkFill, RolloutSample, TokenSpan, VerifyResult


@dataclass
class ForkConfig:
    num_fills: int = 4
    mask_token_id: int = 0


class DiffusionPolicy(ABC):
    """Protocol for dLLM stand-in / real model."""

    mask_token_id: int

    @abstractmethod
    def fill_span(
        self,
        token_ids: list[int],
        span: TokenSpan,
        num_samples: int,
    ) -> list[tuple[list[int], float]]:
        raise NotImplementedError


class TinyStandInPolicy(DiffusionPolicy):
    """CPU tiny model: remasks span and samples replacement tactic tokens (MOCK)."""

    def __init__(
        self,
        vocab_size: int = 128,
        mask_token_id: int = 0,
        seed: int = 0,
        span_fill_token: int = 30,
    ) -> None:
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.span_fill_token = span_fill_token
        self.rng = torch.Generator().manual_seed(seed)

    def fill_span(
        self,
        token_ids: list[int],
        span: TokenSpan,
        num_samples: int,
    ) -> list[tuple[list[int], float]]:
        out: list[tuple[list[int], float]] = []
        for s in range(num_samples):
            new_ids = list(token_ids)
            logprob = 0.0
            for i in range(span.start, span.end + 1):
                tok = self.span_fill_token + (s % 3)
                new_ids[i] = tok
                logprob -= 0.1
            out.append((new_ids, logprob))
        return out


def remask_span(token_ids: list[int], span: TokenSpan, mask_token_id: int) -> list[int]:
    masked = list(token_ids)
    for i in range(span.start, span.end + 1):
        masked[i] = mask_token_id
    return masked


def fork_failed_sample(
    client: LeanClient,
    tokenizer: CharTokenizer,
    policy: TinyStandInPolicy,
    sample: RolloutSample,
    statement: str,
    span: TokenSpan,
    cfg: ForkConfig,
) -> list[ForkFill]:
    """Remask ``span``, draw ``cfg.num_fills`` fills, verify each with Lean."""
    base_ids = sample.token_ids
    _ = remask_span(base_ids, span, cfg.mask_token_id)  # masked state conditions fill
    fills_raw = policy.fill_span(base_ids, span, cfg.num_fills)
    results: list[ForkFill] = []
    for new_ids, lp in fills_raw:
        text = tokenizer.decode(new_ids)
        verify = client.verify_proof(statement, text)
        results.append(ForkFill(token_ids=new_ids, text=text, verify=verify, logprob_span=lp))
    return results


def kernel_located_span(
    verify: VerifyResult,
    text: str,
    tokenizer: CharTokenizer,
) -> TokenSpan | None:
    if verify.success or verify.error_span is None:
        return None
    return tokenizer.char_to_token_span(text, verify.error_span)


def apply_rescue(
    group: list[RolloutSample],
    forks: list[list[ForkFill]],
) -> list[RolloutSample]:
    """If entire group failed, attach passing fork fills as repaired samples."""
    if any(s.verify.success for s in group):
        return group
    repaired = list(group)
    for sample, ff in zip(group, forks):
        for fill in ff:
            if fill.verify.success:
                repaired.append(
                    RolloutSample(
                        prompt=sample.prompt,
                        proof_text=fill.text,
                        token_ids=fill.token_ids,
                        verify=fill.verify,
                        fork_fills=[],
                    )
                )
                break
    return repaired
