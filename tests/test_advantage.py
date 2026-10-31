from kernelfork.advantage import group_relative_advantages, span_advantages_among_fills
from kernelfork.types import ForkFill, VerifyResult


def test_group_relative_zero_std():
    adv = group_relative_advantages([1.0, 1.0, 1.0])
    assert all(abs(a) < 1e-6 for a in adv)


def test_span_advantages_mixed():
    fills = [
        ForkFill([1], "a", VerifyResult(True)),
        ForkFill([2], "b", VerifyResult(False)),
    ]
    adv = span_advantages_among_fills(fills)
    assert len(adv) == 2
    assert adv[0] > adv[1]
