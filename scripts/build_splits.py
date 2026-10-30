#!/usr/bin/env python3
"""Build train/eval splits for RL (statement-level JSON)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="JSONL with {id, statement}")
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--eval-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_lines = args.input.read_text().splitlines()
    lines = [json.loads(line) for line in raw_lines if line.strip()]
    import random

    rng = random.Random(args.seed)
    rng.shuffle(lines)
    n_eval = max(1, int(len(lines) * args.eval_frac))
    eval_set = lines[:n_eval]
    train_set = lines[n_eval:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.jsonl").write_text("\n".join(json.dumps(x) for x in train_set) + "\n")
    (args.output_dir / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in eval_set) + "\n")
    meta = {"train": len(train_set), "eval": len(eval_set), "seed": args.seed, "mock": False}
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
