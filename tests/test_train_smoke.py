import json
from pathlib import Path

import pytest

from kernelfork.fork import ForkConfig
from kernelfork.lean_client import MockLeanClient
from kernelfork.models.tiny import TinyStandInPolicy
from kernelfork.spans import CharTokenizer
from kernelfork.train_grpo import run_grpo_step

FIXTURE = Path("tests/mock_data/problem.json")

pytestmark = pytest.mark.mock


def test_kernelfork_step_rescues_mock_group():
    problem = json.loads(FIXTURE.read_text())
    client = MockLeanClient()
    tokenizer = CharTokenizer(problem.get("vocab", {}))
    good_id = int(problem["vocab"]["good_tactic"])
    policy = TinyStandInPolicy(seed=0, span_fill_token=good_id)
    metrics, detail = run_grpo_step(
        "kernelfork",
        client,
        tokenizer,
        policy,
        problem["statement"],
        problem["candidates"],
        ForkConfig(num_fills=4),
        seed=0,
    )
    assert metrics.mock
    assert metrics.span_forks >= 1
    # Rescue may add passing samples when fork hits good text
    assert detail["rescued_count"] >= 1


def test_diffu_grpo_no_forks():
    problem = json.loads(FIXTURE.read_text())
    client = MockLeanClient()
    tokenizer = CharTokenizer(problem.get("vocab", {}))
    policy = TinyStandInPolicy(seed=0)
    metrics, _ = run_grpo_step(
        "diffu_grpo",
        client,
        tokenizer,
        policy,
        problem["statement"],
        problem["candidates"],
        ForkConfig(num_fills=4),
        seed=0,
    )
    assert metrics.span_forks == 0
