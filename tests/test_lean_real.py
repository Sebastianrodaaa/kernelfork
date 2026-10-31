"""Real Lean 4 tests (no MOCK client)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelfork.fork import ForkConfig, fork_failed_sample, kernel_located_span
from kernelfork.lean_client import CliLeanClient
from kernelfork.models.tiny import TinyStandInPolicy
from kernelfork.spans import CharTokenizer, WholeProofTokenizer
from kernelfork.train_grpo import run_grpo_step
from kernelfork.types import RolloutSample

pytestmark = pytest.mark.real

REAL_FIXTURE = Path("tests/real_data/problem.json")
STMT = "theorem kernelfork_true : True := by"


@pytest.fixture(scope="module")
def lean_client() -> CliLeanClient:
    if not CliLeanClient.is_available():
        pytest.skip("Lean 4 not installed")
    return CliLeanClient()


def test_real_lean_passes_trivial(lean_client: CliLeanClient) -> None:
    r = lean_client.verify_proof(STMT, "  trivial")
    assert r.success is True
    assert r.raw.get("source") == "lean_cli"
    assert "MOCK" not in str(r.raw)


def test_real_lean_fails_unknown_tactic(lean_client: CliLeanClient) -> None:
    r = lean_client.verify_proof(STMT, "  unknown_tactic_kernelfork")
    assert r.success is False
    assert r.error_span is not None
    assert r.error_span.char_start >= 0
    msg = r.error_span.message.lower()
    assert "unknown" in msg or "tactic" in msg


def test_real_kernel_span_maps_to_proof_line(lean_client: CliLeanClient) -> None:
    proof = "  unknown_tactic_kernelfork\n  trivial"
    r = lean_client.verify_proof(STMT, proof)
    assert r.error_span is not None
    tok = CharTokenizer()
    span = tok.char_to_token_span(proof, r.error_span)
    assert span.start <= span.end


def test_real_grpo_rescue_with_whole_proof_tokenizer(lean_client: CliLeanClient) -> None:
    problem = json.loads(REAL_FIXTURE.read_text())
    fail = problem["fail_proof"]
    ok = problem["pass_proof"]
    candidates = [fail, fail, fail, fail]

    tokenizer = WholeProofTokenizer()
    ok_id = tokenizer.encode(ok)[0]

    class RescuePolicy(TinyStandInPolicy):
        def fill_span(self, token_ids, span, num_samples):
            return [([ok_id], -0.05) for _ in range(num_samples)]

    metrics, detail = run_grpo_step(
        "kernelfork",
        lean_client,
        tokenizer,
        RescuePolicy(),
        problem["statement"],
        candidates,
        ForkConfig(num_fills=2),
        seed=0,
    )
    assert metrics.mock is False
    assert metrics.num_pass_initial == 0
    assert metrics.num_pass_after_rescue >= 1
    assert detail["rescued_count"] >= 1


def test_real_fork_verify_passing_fill(lean_client: CliLeanClient) -> None:
    proof = "  unknown_tactic_kernelfork"
    v = lean_client.verify_proof(STMT, proof)
    tokenizer = WholeProofTokenizer()
    ok_id = tokenizer.encode("  trivial")[0]
    sample = RolloutSample(STMT, proof, tokenizer.encode(proof), v)
    span = kernel_located_span(v, proof, tokenizer)
    assert span is not None

    class PassFillPolicy(TinyStandInPolicy):
        def fill_span(self, token_ids, span, num_samples):
            return [([ok_id], -0.1) for _ in range(num_samples)]

    fills = fork_failed_sample(
        lean_client,
        tokenizer,
        PassFillPolicy(),
        sample,
        STMT,
        span,
        ForkConfig(num_fills=2),
    )
    assert any(f.verify.success for f in fills)
    assert all(f.verify.raw.get("source") == "lean_cli" for f in fills)
