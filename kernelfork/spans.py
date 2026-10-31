"""Map Lean error character spans to token spans."""

from __future__ import annotations

from kernelfork.types import LeanErrorSpan, TokenSpan

__all__ = [
    "CharTokenizer",
    "WholeProofTokenizer",
    "random_span_of_length",
]


class CharTokenizer:
    """Simple whitespace/punctuation tokenizer for MOCK pipelines."""

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        self.vocab = vocab or {}

    def encode(self, text: str) -> list[int]:
        if not self.vocab:
            # Stable fake ids from ordinals
            return [hash(tok) % 10000 + 100 for tok in text.split()]
        ids: list[int] = []
        for tok in text.split():
            ids.append(self.vocab.get(tok, self.vocab.get("<unk>", 1)))
        return ids

    def decode(self, ids: list[int]) -> str:
        if not self.vocab:
            return " ".join(f"t{i}" for i in ids)
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)

    def char_to_token_span(self, text: str, error: LeanErrorSpan) -> TokenSpan:
        """Map character span to token span via whitespace token boundaries."""
        tokens = text.split()
        if not tokens:
            return TokenSpan(0, 0)

        char = 0
        start_tok: int | None = None
        end_tok: int | None = None
        for i, tok in enumerate(tokens):
            tok_start = text.find(tok, char)
            tok_end = tok_start + len(tok)
            char = tok_end
            if start_tok is None and tok_end > error.char_start:
                start_tok = i
            if tok_start < error.char_end:
                end_tok = i
        if start_tok is None:
            start_tok = 0
        if end_tok is None:
            end_tok = len(tokens) - 1
        return TokenSpan(start_tok, end_tok)


class WholeProofTokenizer:
    """Maps entire proof strings to single token ids (for real Lean integration tests)."""

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._next = 10
        self._char = CharTokenizer()

    def encode(self, text: str) -> list[int]:
        if text not in self._to_id:
            self._to_id[text] = self._next
            self._next += 1
        return [self._to_id[text]]

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self._to_id.items()}
        return inv.get(ids[0], "")

    def char_to_token_span(self, text: str, error: LeanErrorSpan) -> TokenSpan:
        if len(self.encode(text)) == 1:
            return TokenSpan(0, 0)
        return self._char.char_to_token_span(text, error)


def random_span_of_length(num_tokens: int, length: int, rng) -> TokenSpan:
    """Pick a random contiguous span with fixed length (ablation baseline)."""
    length = max(1, min(length, num_tokens))
    if num_tokens <= length:
        return TokenSpan(0, num_tokens - 1)
    start = int(rng.integers(0, num_tokens - length + 1))
    return TokenSpan(start, start + length - 1)
