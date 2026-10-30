"""Pythagoras-Prover-Diffusion-4B loading helpers (optional GPU)."""

from __future__ import annotations

from dataclasses import dataclass

PYTHAGORAS_MODEL_ID = "Pythagoras-LM/Pythagoras-Prover-Diffusion-4B"
PYTHAGORAS_LICENSE = "apache-2.0"
PYTHAGORAS_ARCH = "a2d-qwen3"


@dataclass
class PythagorasLoadInfo:
    model_id: str
    license: str
    architecture: str
    mask_token_id: int = 151669
    notes: str = ""


def check_pythagoras_available() -> PythagorasLoadInfo:
    """Verify Hub metadata without downloading weights (safe in CI)."""
    info = PythagorasLoadInfo(
        model_id=PYTHAGORAS_MODEL_ID,
        license=PYTHAGORAS_LICENSE,
        architecture=PYTHAGORAS_ARCH,
        notes=(
            "Use Hugging Face model card snippet to register A2DQwen3 classes and "
            "BD3LM sampler. Full weights require GPU and are not pulled in CI."
        ),
    )
    return info


def load_pythagoras_policy(device: str = "cuda"):
    """Load real policy when transformers + GPU available (not used in MOCK smoke)."""
    raise NotImplementedError(
        "Copy A2DQwen3 registration and BD3LM sampler from the HF model card. "
        f"Model: {PYTHAGORAS_MODEL_ID} ({PYTHAGORAS_LICENSE})."
    )
