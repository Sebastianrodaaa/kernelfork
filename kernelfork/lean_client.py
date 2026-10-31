"""Lean verification client (mock + REPL stub)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kernelfork.types import LeanErrorSpan, VerifyResult

TACTIC_SPLIT_RE = re.compile(r"\n(?=\s*(?:by\s*\n|·|[-|]))")


class LeanClient(ABC):
    @abstractmethod
    def verify_proof(self, statement: str, proof: str) -> VerifyResult:
        raise NotImplementedError


class MockLeanClient(LeanClient):
    """Deterministic Lean stand-in for CPU smoke tests (MOCK).

    Reads ``tests/mock_data/lean_responses.json`` keyed by proof hash / tactic text.
    """

    def __init__(self, fixture_path: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[1]
        path = fixture_path or root / "tests" / "mock_data" / "lean_responses.json"
        with path.open() as f:
            self._rules: dict[str, Any] = json.load(f)

    def verify_proof(self, statement: str, proof: str) -> VerifyResult:
        key = proof.strip()
        entry = self._rules.get(key)
        if entry is None:
            # Default: fail on tactic containing "WRONG"
            if "WRONG" in proof:
                span = self._first_wrong_span(proof)
                return VerifyResult(
                    success=False,
                    error_span=span,
                    raw={"source": "MOCK", "reason": "contains WRONG"},
                )
            return VerifyResult(success=True, raw={"source": "MOCK"})

        if entry.get("success"):
            return VerifyResult(success=True, raw={"source": "MOCK", **entry})

        es = entry["error_span"]
        span = LeanErrorSpan(
            tactic_index=es["tactic_index"],
            char_start=es["char_start"],
            char_end=es["char_end"],
            message=es.get("message", "MOCK error"),
            goal_state=es.get("goal_state"),
        )
        return VerifyResult(success=False, error_span=span, raw={"source": "MOCK", **entry})

    @staticmethod
    def _first_wrong_span(proof: str) -> LeanErrorSpan:
        idx = proof.index("WRONG")
        # Expand to line boundaries for tactic-like span
        line_start = proof.rfind("\n", 0, idx) + 1
        line_end = proof.find("\n", idx)
        if line_end == -1:
            line_end = len(proof)
        return LeanErrorSpan(
            tactic_index=0,
            char_start=line_start,
            char_end=line_end,
            message="unknown identifier 'WRONG'",
            goal_state="⊢ MOCK_goal",
        )


class ReplLeanClient(LeanClient):
    """Optional Lean 4 REPL client (skipped in CI when ``lean`` is missing)."""

    def __init__(self, repl_cmd: list[str] | None = None, timeout_s: float = 30.0) -> None:
        self.repl_cmd = repl_cmd or ["lean", "--version"]
        self.timeout_s = timeout_s

    @staticmethod
    def is_available() -> bool:
        try:
            subprocess.run(
                ["lean", "--version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.SubprocessError):
            return False

    def verify_proof(self, statement: str, proof: str) -> VerifyResult:
        if not self.is_available():
            raise RuntimeError("Lean 4 not available; use MockLeanClient or install Lean.")

        # Placeholder: real Kimina/LeanDojo REPL integration belongs here.
        env = os.environ.get("KERNELFORK_LEAN_REPL")
        if not env:
            raise NotImplementedError(
                "ReplLeanClient requires KERNELFORK_LEAN_REPL command configuration."
            )
        proc = subprocess.run(
            [env, statement, proof],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        payload = json.loads(proc.stdout or "{}")
        if payload.get("success"):
            return VerifyResult(success=True, raw=payload)
        es = payload.get("error_span", {})
        span = LeanErrorSpan(
            tactic_index=int(es.get("tactic_index", 0)),
            char_start=int(es["char_start"]),
            char_end=int(es["char_end"]),
            message=str(es.get("message", "Lean error")),
            goal_state=es.get("goal_state"),
        )
        return VerifyResult(success=False, error_span=span, raw=payload)


def compile_first_error_span(proof: str, diagnostics: list[dict[str, Any]]) -> LeanErrorSpan | None:
    """Map Lean diagnostic JSON to a tactic character span (helper for REPL parsers)."""
    if not diagnostics:
        return None
    d0 = diagnostics[0]
    pos = d0.get("pos", {})
    return LeanErrorSpan(
        tactic_index=int(d0.get("tactic_index", 0)),
        char_start=int(pos.get("start", 0)),
        char_end=int(pos.get("end", len(proof))),
        message=str(d0.get("message", "")),
        goal_state=d0.get("goal_state"),
    )
