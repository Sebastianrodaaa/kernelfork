import pytest

from kernelfork.lean_client import MockLeanClient
from kernelfork.spans import CharTokenizer
from kernelfork.types import LeanErrorSpan

pytestmark = pytest.mark.mock


def test_char_to_token_span():
    text = "by\n  WRONG_tactic\n  ring"
    err = LeanErrorSpan(0, text.index("WRONG"), text.index("WRONG") + 5, "err")
    tok = CharTokenizer()
    span = tok.char_to_token_span(text, err)
    assert span.start <= span.end


def test_mock_lean_wrong():
    client = MockLeanClient()
    r = client.verify_proof("stmt", "by WRONG x")
    assert not r.success
    assert r.error_span is not None
