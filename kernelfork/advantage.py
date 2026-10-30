"""Span-level and sequence-level GRPO advantages."""

from __future__ import annotations

import numpy as np

from kernelfork.types import ForkFill, TokenSpan


def reward_from_verify(success: bool) -> float:
    return 1.0 if success else 0.0


def group_relative_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """Standard GRPO group-relative normalization."""
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    if std < eps:
        return [0.0 for _ in rewards]
    return ((arr - mean) / (std + eps)).tolist()


def span_advantages_among_fills(fills: list[ForkFill]) -> list[float]:
    rewards = [reward_from_verify(f.verify.success) for f in fills]
    return group_relative_advantages(rewards)


def combined_token_advantages(
    seq_adv: float,
    span_adv: float,
    num_tokens: int,
    span: TokenSpan,
) -> list[float]:
    """Sequence advantage on all tokens; span tokens add ``span_adv``."""
    adv = [seq_adv] * num_tokens
    for i in range(span.start, min(span.end + 1, num_tokens)):
        adv[i] = seq_adv + span_adv
    return adv


def masked_diffusion_logprob_weight(logprob_span: float) -> float:
    """Placeholder weight for span likelihood under masked diffusion (MOCK stand-in)."""
    return float(np.exp(logprob_span))
