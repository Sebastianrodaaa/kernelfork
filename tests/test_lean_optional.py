import pytest

from kernelfork.lean_client import CliLeanClient, ReplLeanClient

pytestmark = pytest.mark.real


@pytest.mark.lean
def test_lean_cli_available():
    if not CliLeanClient.is_available():
        pytest.skip("Lean 4 not installed")
    client = CliLeanClient()
    ok = client.verify_proof("theorem t : True := by", "  trivial")
    assert ok.success


def test_repl_delegates_to_cli_without_env():
    if not ReplLeanClient.is_available():
        pytest.skip("Lean 4 not installed")
    client = ReplLeanClient()
    ok = client.verify_proof("theorem t : True := by", "  trivial")
    assert ok.success
    assert ok.raw.get("source") == "lean_cli"
