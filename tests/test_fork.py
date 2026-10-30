from kernelfork.fork import ForkConfig, fork_failed_sample, kernel_located_span
from kernelfork.lean_client import MockLeanClient
from kernelfork.models.tiny import TinyStandInPolicy
from kernelfork.spans import CharTokenizer
from kernelfork.types import RolloutSample, VerifyResult


def test_kernel_fork_produces_passing_fill():
    client = MockLeanClient()
    tokenizer = CharTokenizer({"good_tactic": 30, "ring": 40, "by": 10, "<unk>": 1})
    policy = TinyStandInPolicy(seed=0)
    proof = "by good_tactic ring"
    sample = RolloutSample("stmt", proof, tokenizer.encode(proof), VerifyResult(False))
    # Force fail first
    sample.verify = client.verify_proof("stmt", "by WRONG_tactic ring")
    sample.proof_text = "by WRONG_tactic ring"
    sample.token_ids = tokenizer.encode(sample.proof_text)
    span = kernel_located_span(sample.verify, sample.proof_text, tokenizer)
    assert span is not None
    fills = fork_failed_sample(
        client,
        tokenizer,
        policy,
        sample,
        "stmt",
        span,
        ForkConfig(num_fills=4),
    )
    assert len(fills) == 4
