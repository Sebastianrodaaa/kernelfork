# kernelfork — research report (v0.1 harness)

**Status:** implementation + MOCK CPU validation complete; **full GPU + Lean experiment pending.**

## Motivation

GRPO on whole-proof diffusion LM provers assigns one scalar reward per sample. When every sample in a group fails, the group contributes **no learning signal**. Lean already localizes the **first failing tactic** and its goal state. Masked diffusion models can **remask and resample** a contiguous span while keeping both prefix and suffix fixed — something autoregressive provers cannot do in-place.

**kernelfork** uses the kernel-reported span to draw **B** counterfactual fills, verifies each with Lean, and uses outcomes for:

1. **Span-level advantages** among fills (GRPO-style), applied to span token positions via masked-diffusion likelihood weights.
2. **Rescue** of all-fail groups when any fork verifies.

The falsifiable claim (from the project brief): verifier-located forks beat matched-compute **diffu-GRPO**, **DiSPO-style** branching, and **random-span** forks of equal length; random-span parity would refute the idea.

## Method (implemented)

| Component | Module | Notes |
|-----------|--------|-------|
| Lean verify + first error span | `kernelfork/lean_client.py` | `MockLeanClient` for CI; `ReplLeanClient` stub for real REPL |
| Char/token span map | `kernelfork/spans.py` | MOCK whitespace tokenizer; swap for prover tokenizer in v1 |
| In-place remask + fill | `kernelfork/fork.py` | `TinyStandInPolicy` on CPU; Pythagoras loader stub for GPU |
| Advantages | `kernelfork/advantage.py` | Group-relative + span-local combined token advantages |
| Training step | `kernelfork/train_grpo.py` | Arms: `kernelfork`, `diffu_grpo`, `dispo_branch`, `random_span` |
| Baselines | `baselines/` | Thin CLIs delegating to `train_grpo` |
| Splits | `scripts/build_splits.py` | Statement-level JSONL split |

## Related work

See [RELATED_WORK.md](./RELATED_WORK.md). No prior work found that combines **Lean-located** spans, **two-sided** in-place remasking, and **GRPO** fork advantages on a dLLM prover. Closest: Process-Verified RL (AR + tactic shaping), DiSPO/ForkGRPO (non-verifier branch points), verimask (inference remasking, frozen denoiser).

## Design decisions

1. **MOCK-first CI:** All default paths use `tests/mock_data/` and set `"mock": true` in metrics.
2. **Pythagoras policy:** Verified Hub license **Apache-2.0**, architecture **`a2d-qwen3`**, `mask_token_id=151669`. We do **not** download 4B weights in CI; loading code is documented in `kernelfork/models/pythagoras.py` and the HF model card.
3. **Fallback:** Brief specifies LLaDA-8B-Instruct + light SFT if Pythagoras loading fails on a given machine.
4. **Key ablation:** `random_span` arm remasks equal-length spans not tied to Lean diagnostics.

## Experiment plan (not yet executed here)

| Setting | Value |
|---------|--------|
| Policy | Pythagoras-Prover-Diffusion-4B |
| Verifier | Lean 4 + Mathlib (pinned to model training) |
| Train | Lean-verified open statements (e.g. Pythagoras SFT set) |
| Eval | miniF2F-test (+ ProofNet if time) |
| Arms | diffu-GRPO, DiSPO-branch, random-span, kernelfork (B∈{2,4}) |
| Seeds | 3 × ~200 RL steps, matched Lean calls |

**Metrics:** pass@1, pass@8, fraction of groups with nonzero advantage, rescue rate, span length vs repair rate, Lean calls, GPU-hours.

## What was actually run (this environment)

| Run | Environment | Output |
|-----|-------------|--------|
| `make smoke` | CPU, MOCK fixtures | `results/MOCK/smoke_step.json` |
| `make repro` | CPU, MOCK, 3 seeds × 4 arms | `results/MOCK/step_*.json` |
| Pytest | GitHub Actions–style, `-m "not lean"` | unit + integration |

**No** real pass@k numbers, **no** Pythagoras GPU training, **no** Mathlib proof statistics — by design until GPU repro.

## Results (MOCK only)

Example smoke metrics shape (values from MOCK harness, **not** benchmarks):

```json
{
  "arm": "kernelfork",
  "num_pass_initial": 0,
  "num_pass_after_rescue": ">0 when forks hit good_tactic",
  "mock": true
}
```

Do **not** compare MOCK JSON to miniF2F targets in the brief.

## Limitations

- `ReplLeanClient` is a configuration stub; production use needs Kimina/LeanDojo-style REPL wiring.
- Token/char alignment is simplified; production must use the prover tokenizer and Lean column maps (cf. verimask parser notes in brief).
- `TinyStandInPolicy` is not a diffusion forward pass; real training must plug BD3LM sampling from Pythagoras.
- Span likelihood uses a placeholder weight (`advantage.masked_diffusion_logprob_weight`).

## Next steps

1. Integrate HF A2DQwen3 + BD3LM sampler from Pythagoras model card.
2. Wire Lean REPL with pinned Mathlib; port/error-span parser from verimask where applicable.
3. Run matched-compute v1 experiment; commit **real** JSON under `results/<run>/`.
4. Plot pass@k and rescue curves into README via `results/*.json` (never fabricate tables).

## License

Code: **MIT** (repository root). Third-party: Pythagoras-Prover-Diffusion-4B **Apache-2.0**, Mathlib **Apache-2.0**, dataset licenses per upstream Hugging Face dataset cards.
