from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LeanErrorSpan:
    """Character span of the first failing tactic in a Lean proof attempt."""

    tactic_index: int
    char_start: int
    char_end: int
    message: str
    goal_state: str | None = None


@dataclass
class TokenSpan:
    """Inclusive token index range [start, end] in a tokenized proof."""

    start: int
    end: int

    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class VerifyResult:
    success: bool
    error_span: LeanErrorSpan | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForkFill:
    token_ids: list[int]
    text: str
    verify: VerifyResult
    logprob_span: float = 0.0


@dataclass
class RolloutSample:
    prompt: str
    proof_text: str
    token_ids: list[int]
    verify: VerifyResult
    fork_fills: list[ForkFill] = field(default_factory=list)
