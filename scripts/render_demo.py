#!/usr/bin/env python3
"""Render README demo GIF from MOCK smoke pipeline (matplotlib animation)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "mock_data" / "problem.json"
OUT = ROOT / "docs" / "demo.gif"


def main() -> None:
    problem = json.loads(FIXTURE.read_text())
    failing = problem["candidates"][0]
    fixed = "by\n  good_tactic\n  ring"
    err_line = "by\n  WRONG_tactic\n  ring"
    wrong_idx = failing.index("WRONG")

    frames = [
        ("1. Rollout (MOCK)", failing, None, "Lean: fail — unknown WRONG"),
        ("2. Kernel span", failing, (wrong_idx, wrong_idx + 12), "Remask tactic span"),
        ("3. Fork fill B=2", fixed.replace("good_tactic", "fork_a"), None, "Verify fork A…"),
        ("4. Fork accepted", fixed, None, "Lean: pass (rescue)"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.axis("off")
    ta = ax.transAxes

    def draw_frame(i):
        ax.clear()
        ax.axis("off")
        title, text, highlight, footer = frames[i % len(frames)]
        ax.text(0.02, 0.92, title, color="#cdd6f4", fontsize=14, weight="bold", transform=ta)
        ax.text(0.02, 0.78, "MOCK problem", color="#a6adc8", fontsize=10, transform=ta)
        y = 0.62
        for line in text.split("\n"):
            color = "#f38ba8" if "WRONG" in line else "#a6e3a1"
            ax.text(0.04, y, line, color=color, fontsize=11, family="monospace", transform=ta)
            y -= 0.12
        if highlight:
            ax.add_patch(
                Rectangle(
                    (0.03, 0.38),
                    0.94,
                    0.14,
                    fill=False,
                    edgecolor="#fab387",
                    linewidth=2,
                    transform=ta,
                )
            )
        ax.text(0.02, 0.08, footer, color="#89b4fa", fontsize=10, transform=ta)
        ax.text(0.02, 0.02, err_line[:40] + "...", color="#585b70", fontsize=8, transform=ta)

    n_frames = len(frames) * 3
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_frames, interval=900, repeat=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer="pillow", fps=1.2, dpi=100)
    plt.close()
    print(f"Wrote {OUT} (MOCK)")


if __name__ == "__main__":
    main()
