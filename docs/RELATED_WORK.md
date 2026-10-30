# Related work (checked 2026-10-30)

This note compares **kernelfork** — Lean-kernel-located, in-place span resampling during GRPO on masked diffusion LM (dLLM) theorem provers — to the closest prior art on arXiv, Hugging Face, and GitHub. Search queries included: `dllm lean`, `diffusion lean prover`, `diffusion theorem proving lean`, `dllm grpo credit`, `verifier span resampling diffusion RL`, and manual review of papers cited in the project brief.

## Summary

| Work | Verifier | dLLM | In-place remask (prefix+suffix fixed) | Fork location | RL credit on fork fills |
|------|----------|------|----------------------------------------|---------------|-------------------------|
| Process-Verified RL (Lean) [2606.20068] | Lean (tactic-level) | No (AR provers) | No | First-error tactic (reward only) | Process shaping, no counterfactual resampling |
| DiSPO [2602.06462] | Terminal task reward | Yes (MDLM) | Partial (only still-masked positions) | Model rollout state, not verifier | Yes, among branch fillings |
| ForkGRPO [2610.19408] | Terminal reward | Yes | No (prefix-only continuations) | High-entropy intermediate states | Segment + continuation GRPO |
| DTCA [2610.35201] | None (confidence dynamics) | Yes | No | N/A | Token weights from denoising |
| DACA-GRPO [2605.16342] | None | Yes | No | N/A | Denoising-step / stratified likelihood |
| VICT [2608.28128] | Lean atoms → actions | No (AR agents) | No | Dependency graph | Trace-based credit |
| Diffusion-Proof [2606.19315] | Compile/check | Yes (prover) | Inference refinement | N/A | SFT-focused in release |
| Pythagoras-Prover-Diffusion [2606.12594] | Lean compile reward | Yes | Inference-time diffusion decode | N/A | RL with compile reward (no span forks described) |
| verimask [Sebastianrodaaa/verimask] | Lean (inference) | Yes | Yes (inference remask policy) | Learned remask head | Inference only; frozen denoiser |
| **kernelfork (this repo)** | **Lean first failing tactic** | **Yes** | **Yes (reopen committed span)** | **Kernel error span** | **Span-level GRPO among fork fills + rescue** |

**Conclusion:** We did **not** find a published paper or public repository that combines (i) GRPO on a dLLM Lean prover, (ii) remasking **only** the Lean-reported first failing tactic span while keeping prefix and suffix fixed, (iii) drawing multiple in-place counterfactual fills verified by Lean, and (iv) using those fills for span-local advantages under the masked-diffusion likelihood **and** for rescuing all-fail GRPO groups. The combination of **verifier-located** forks with **two-sided** conditioning during **RL** (not inference-only remasking) is the claimed novelty.

**Near-miss (important):** DiSPO and ForkGRPO already resample under diffusion RL, but branch points are not Lean-located and they do not reopen committed tokens around a fixed suffix. Process-Verified RL uses Lean’s first error for **credit shaping on AR models**, not in-place dLLM forks. **verimask** is the closest *mechanism* (Lean-guided remasking on a dLLM) but targets **inference** with a **frozen** denoiser rather than training the denoiser via kernel forks.

## Papers (brief)

### Process-Verified Reinforcement Learning for Theorem Proving via Lean (arXiv:2606.20068)

Maps Lean elaboration to tactic-level scores with first-error propagation and first-token credit on **autoregressive** provers (STP-Lean, DeepSeek-Prover-V1.5). Dense verifier signal without counterfactual span resampling.

### DiSPO — Diffusion-State Policy Optimization (arXiv:2602.06462)

Plug-in for masked diffusion PO: at selected **intermediate masked states** during rollout, resample fillings for **currently masked** positions from cached logits; score branched completions; update newly filled tokens. Branch timing is **not** verifier-driven; **committed** tokens are not reopened.

### ForkGRPO (arXiv:2610.19408)

Forks **continuations** from high-entropy states on a left-to-right reference trajectory; **prefix reuse** only. Segment credit via Monte Carlo state-value differences + group-relative continuation advantages. No Lean location, no suffix conditioning.

### DTCA (arXiv:2610.35201)

Token credit from improvement in confidence on remaining masked positions after each commit; POS aggregation and entropy calibration. **No external verifier.**

### DACA-GRPO (arXiv:2605.16342)

Denoising progress scores + stratified masking likelihood for GRPO on dLLMs. **No external verifier.**

### VICT (arXiv:2608.28128)

Traces verifier atoms to agent actions via dependency edges for **AR** agents. No in-place diffusion resampling.

### Diffusion-Proof (arXiv:2606.19315)

Diffusion-based theorem proving line; training emphasis on SFT / diffusion decoding rather than Lean-located RL forks.

### Pythagoras-Prover-Diffusion (arXiv:2606.12594)

First open diffusion Lean prover family (A2D-Qwen3, BD3LM sampler). Hugging Face checkpoint `Pythagoras-LM/Pythagoras-Prover-Diffusion-4B` is **Apache-2.0** with custom `a2d-qwen3` architecture and in-card loading/sampling code. Planned v1 policy for kernelfork experiments; this repo ships a **tiny stand-in** for CPU smoke tests.

### LeanGRPO (arXiv:2609.03528) — orthogonal

Systems-level **recomputation** savings for diffusion RL (Retain/Reweight schedules), not verifier-located span forks.

### SHEAR (arXiv:2604.23318) — orthogonal

Span-level advantages from hidden-state Wasserstein distance between correct/incorrect rollouts in RLVR; no Lean, no remasking.

## GitHub / Hugging Face sweep (2026-10-30)

- **verimask** (`github.com/Sebastianrodaaa/verimask`, per brief): Lean-informed **inference** remasking with frozen denoiser — closest codebase theme; training-time kernel forks are **not** described there.
- **Pythagoras-LM/Pythagoras-Prover** — prover weights and BD3LM sampler; no kernelfork-style RL.
- **ZJU-OmniAI/DTCA**, **daioba/dispo** (expected) — dLLM RL credit, no Lean span forks.
- Unrelated homonyms: **VeriMask_Designs** (N95 mask hardware), **remdm** / **RemeDi** (general remasking samplers, not Lean RL credit).

## Implication for kernelfork

If **random-span** forks (equal length, no Lean location) match **kernel-located** forks, the hypothesis in the brief fails. The key ablation is implemented in `baselines/random_span.py`.

## References (URLs)

- Process-Verified RL: https://arxiv.org/abs/2606.20068  
- DiSPO: https://arxiv.org/abs/2602.06462  
- ForkGRPO: https://arxiv.org/abs/2610.19408  
- DTCA: https://arxiv.org/abs/2610.35201  
- DACA-GRPO: https://arxiv.org/abs/2605.16342  
- VICT: https://arxiv.org/abs/2608.28128  
- Diffusion-Proof: https://arxiv.org/abs/2606.19315  
- Pythagoras-Prover-Diffusion: https://arxiv.org/abs/2606.12594  
- Pythagoras-Prover-Diffusion-4B: https://huggingface.co/Pythagoras-LM/Pythagoras-Prover-Diffusion-4B  
- verimask: https://github.com/Sebastianrodaaa/verimask  
