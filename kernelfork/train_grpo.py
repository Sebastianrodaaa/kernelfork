"""GRPO training step with optional kernelfork forks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from kernelfork.advantage import (
    combined_token_advantages,
    group_relative_advantages,
    reward_from_verify,
    span_advantages_among_fills,
)
from kernelfork.fork import (
    ForkConfig,
    apply_rescue,
    fork_failed_sample,
    kernel_located_span,
)
from kernelfork.lean_client import LeanClient, MockLeanClient
from kernelfork.models.tiny import TinyStandInPolicy
from kernelfork.spans import CharTokenizer, random_span_of_length
from kernelfork.types import RolloutSample

ArmName = Literal["kernelfork", "diffu_grpo", "dispo_branch", "random_span"]


@dataclass
class StepMetrics:
    arm: str
    group_size: int
    num_pass_initial: int
    num_pass_after_rescue: int
    nonzero_advantage_fraction: float
    span_forks: int
    mock: bool = True


def load_problem(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def rollout_group(
    client: LeanClient,
    tokenizer: CharTokenizer,
    policy: TinyStandInPolicy,
    statement: str,
    candidates: list[str],
) -> list[RolloutSample]:
    samples: list[RolloutSample] = []
    for proof in candidates:
        ids = tokenizer.encode(proof)
        verify = client.verify_proof(statement, proof)
        samples.append(
            RolloutSample(
                prompt=statement,
                proof_text=proof,
                token_ids=ids,
                verify=verify,
            )
        )
    return samples


def run_grpo_step(
    arm: ArmName,
    client: LeanClient,
    tokenizer: CharTokenizer,
    policy: TinyStandInPolicy,
    statement: str,
    candidates: list[str],
    fork_cfg: ForkConfig,
    seed: int = 0,
) -> tuple[StepMetrics, dict]:
    group = rollout_group(client, tokenizer, policy, statement, candidates)
    rewards = [reward_from_verify(s.verify.success) for s in group]
    seq_advs = group_relative_advantages(rewards)

    span_forks = 0
    all_forks: list[list] = []
    token_grad_signals: list[dict] = []

    for idx, sample in enumerate(group):
        seq_adv = seq_advs[idx]
        if sample.verify.success:
            token_grad_signals.append({"sample": idx, "seq_adv": seq_adv, "span_adv": 0.0})
            all_forks.append([])
            continue

        span = None
        if arm == "kernelfork":
            span = kernel_located_span(sample.verify, sample.proof_text, tokenizer)
        elif arm == "random_span":
            err = sample.verify.error_span
            length = 3
            if err is not None:
                ts = tokenizer.char_to_token_span(sample.proof_text, err)
                length = ts.length()
            span = random_span_of_length(
                len(sample.token_ids),
                length,
                np.random.default_rng(seed + idx),
            )
        elif arm == "dispo_branch":
            # DiSPO-style: branch at an intermediate masked state (MOCK: random equal-length span).
            n = len(sample.token_ids)
            err = sample.verify.error_span
            length = 1
            if err is not None:
                length = tokenizer.char_to_token_span(sample.proof_text, err).length()
            span = random_span_of_length(n, length, np.random.default_rng(seed + idx + 17))
        elif arm == "diffu_grpo":
            token_grad_signals.append({"sample": idx, "seq_adv": seq_adv, "span_adv": 0.0})
            all_forks.append([])
            continue

        if span is None:
            token_grad_signals.append({"sample": idx, "seq_adv": seq_adv, "span_adv": 0.0})
            all_forks.append([])
            continue

        fills = fork_failed_sample(client, tokenizer, policy, sample, statement, span, fork_cfg)
        span_forks += 1
        sample.fork_fills = fills
        all_forks.append(fills)
        span_advs = span_advantages_among_fills(fills)
        # Use mean span advantage for the parent sample's span tokens
        mean_span_adv = sum(span_advs) / len(span_advs) if span_advs else 0.0
        combined = combined_token_advantages(seq_adv, mean_span_adv, len(sample.token_ids), span)
        token_grad_signals.append(
            {
                "sample": idx,
                "seq_adv": seq_adv,
                "span_adv": mean_span_adv,
                "token_adv_sum": sum(combined),
                "span": asdict(span),
            }
        )

    rescued = apply_rescue(group, all_forks)
    pass_after = sum(1 for s in rescued if s.verify.success)
    nonzero = sum(1 for a in seq_advs if abs(a) > 1e-8) / max(len(seq_advs), 1)

    using_mock = isinstance(client, MockLeanClient)
    metrics = StepMetrics(
        arm=arm,
        group_size=len(group),
        num_pass_initial=sum(rewards),
        num_pass_after_rescue=pass_after,
        nonzero_advantage_fraction=nonzero,
        span_forks=span_forks,
        mock=using_mock,
    )
    detail = {
        "metrics": asdict(metrics),
        "token_grad_signals": token_grad_signals,
        "rescued_count": pass_after - sum(rewards),
    }
    return metrics, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GRPO step (MOCK or real).")
    parser.add_argument(
        "--arm",
        default="kernelfork",
        choices=["kernelfork", "diffu_grpo", "dispo_branch", "random_span"],
    )
    parser.add_argument("--fixture", type=Path, default=Path("tests/mock_data/problem.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forks", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    problem = load_problem(args.fixture)
    client = MockLeanClient()
    tokenizer = CharTokenizer(problem.get("vocab", {}))
    vocab = problem.get("vocab", {})
    good_id = int(vocab.get("good_tactic", 30))
    policy = TinyStandInPolicy(seed=args.seed, span_fill_token=good_id)
    fork_cfg = ForkConfig(num_fills=args.forks, mask_token_id=0)

    metrics, detail = run_grpo_step(
        args.arm,
        client,
        tokenizer,
        policy,
        problem["statement"],
        problem["candidates"],
        fork_cfg,
        seed=args.seed,
    )
    out = {"metrics": asdict(metrics), **detail}
    text = json.dumps(out, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
