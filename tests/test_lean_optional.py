import pytest

from kernelfork.lean_client import ReplLeanClient


@pytest.mark.lean
def test_lean_repl_if_installed():
    if not ReplLeanClient.is_available():
        pytest.skip("Lean 4 not installed")
    client = ReplLeanClient()
    # Real REPL not configured in CI
    with pytest.raises((NotImplementedError, RuntimeError)):
        client.verify_proof("theorem t : True := by", " trivial")
