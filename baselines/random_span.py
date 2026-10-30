"""Random-span fork ablation (equal length, no Lean location)."""

from kernelfork.train_grpo import main

if __name__ == "__main__":
    import sys

    if "--arm" not in sys.argv:
        sys.argv.extend(["--arm", "random_span"])
    main()
