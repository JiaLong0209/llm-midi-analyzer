"""
config.py — SOLID-Compliant Configuration System
=================================================
S — Single Responsibility: Each dataclass owns exactly one concern area.
O — Open/Closed: New configs can be added as new dataclasses without modifying existing ones.
L — Liskov Substitution: All config classes implement `IConfig` protocol.
I — Interface Segregation: Trainer only sees TrainingConfig; model only sees ModelConfig.
D — Dependency Inversion: Trainer accepts an abstract IConfig, not a concrete dict.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Protocol, runtime_checkable
import json


# ────────────────────────────────────────────────────────────────
# Interface (I / D in SOLID)
# ────────────────────────────────────────────────────────────────
@runtime_checkable
class IConfig(Protocol):
    """Marker protocol — every config section must be serialisable."""
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "IConfig": ...


# ────────────────────────────────────────────────────────────────
# Data Configs (S — single responsibility each)
# ────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    lmd_dir: str = "data/lmd_matched"
    h5_cache: str = "data/lmd_octuple_cache.h5"
    seq_len: int = 128
    # Set to None for full dataset, or an integer for a quick smoke test
    max_files: int | None = 200
    # Toggle between tokenization strategies:
    #   "miditok_5d"  — miditok.Octuple → (Pitch, Pos, Bar, Vel, Dur)    → input_dim=5
    #   "octuple_8d"  — custom extractor → (Bar, Pos, Prog, Pitch, Dur, Vel, TimeSig, Tempo) → input_dim=8
    token_mode: str = "miditok_5d"

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig": return cls(**d)


TOKEN_MODE_DIM: dict[str, int] = {
    "miditok_5d": 5,
    "octuple_8d": 8,
}

def resolve_input_dim(token_mode: str) -> int:
    """Returns the feature dimension for a given token_mode."""
    if token_mode not in TOKEN_MODE_DIM:
        raise ValueError(f"Unknown token_mode: {token_mode!r}. Choose from {list(TOKEN_MODE_DIM)}.")
    return TOKEN_MODE_DIM[token_mode]


# ────────────────────────────────────────────────────────────────
# Model Config (S)
# ────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    # input_dim is auto-resolved from DataConfig.token_mode at runtime;
    # you can still override it manually if needed.
    input_dim: int = -1          # -1 = auto-resolve via resolve_input_dim()
    hidden_dim: int = 256
    codebook_size: int = 512
    commitment_cost: float = 1.0
    # "hierarchical" | "vanilla"
    variant: str = "hierarchical"

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig": return cls(**d)


# ────────────────────────────────────────────────────────────────
# Training Config (S)
# ────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 4
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"   # "auto" resolves to cuda if available

    def resolved_device(self) -> str:
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig": return cls(**d)


# ────────────────────────────────────────────────────────────────
# Adapter Config (MIDI-to-LLM projection ablation toggle)
# ────────────────────────────────────────────────────────────────
@dataclass
class AdapterConfig:
    # "direct"   → DirectMLPAdapter (Path A baseline)
    # "vqvae"    → CrossAttentionAdapter (Path B proposed)
    # "musicbert"→ MusicBERTAdapter (Path C pretrained encoder)
    projection_mode: str = "direct"
    d_llm: int = 2048                      # Llama 3.2 1B hidden dim
    d_vq: int = 256                        # Must match VQ-VAE hidden_dim (e.g. best8d)
    num_heads: int = 8                     # Cross-attention heads
    input_dim: int = 8                     # OctupleMIDI input features
    codebook_size: int = 512               # Must match VQ-VAE codebook_size
    vqvae_checkpoint: str = None           # Path to .pt file (Path B only)
    musicbert_model_path: str = "roberta-base" # HF path or local dir (Path C only)
    vqvae_min_codebook_usage: float = 0.10 # Health check threshold
    freeze_vqvae: bool = True              # Only train adapter weights
    qlora_r: int = 16                      # QLoRA rank
    qlora_alpha: int = 32                  # QLoRA alpha
    llm_model_path: str = "models/MIDI-LLM"
    theory_context: str = ""              # RAG placeholder <theory_context>
    seq_len: int = 128                   # Default sequence length
    use_unsloth: bool = False            # Use Unsloth for 2x faster training

    def to_dict(self) -> dict: return asdict(self)
    @classmethod
    def from_dict(cls, d: dict) -> "AdapterConfig": return cls(**d)


# ────────────────────────────────────────────────────────────────
# Root Config — composes the above (O — open for extension)
# ────────────────────────────────────────────────────────────────
@dataclass
class OmniConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return {
            "data": self.data.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OmniConfig":
        return cls(
            data=DataConfig.from_dict(d.get("data", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Config saved → {path}")

    @classmethod
    def load(cls, path: str) -> "OmniConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ────────────────────────────────────────────────────────────────
# Pre-baked profiles (Open/Closed — add new ones without changes)
# ────────────────────────────────────────────────────────────────
def smoke_test_config() -> OmniConfig:
    """Tiny config for CI / correctness checks."""
    return OmniConfig(
        data=DataConfig(max_files=100, seq_len=64, token_mode="miditok_5d"),
        model=ModelConfig(hidden_dim=64, codebook_size=64),
        training=TrainingConfig(batch_size=8, num_epochs=3, save_every=1),
    )

def full_train_config() -> OmniConfig:
    """Production config for full LMD training."""
    return OmniConfig(
        data=DataConfig(max_files=None, token_mode="miditok_5d"),
        model=ModelConfig(),
        training=TrainingConfig(num_epochs=20),
    )

def smoke_test_config_8d() -> OmniConfig:
    """Smoke test using 8-dim OctupleMIDI extractor."""
    return OmniConfig(
        data=DataConfig(max_files=100, seq_len=64, token_mode="octuple_8d",
                        h5_cache="data/lmd_octuple8d_cache.h5"),
        model=ModelConfig(hidden_dim=64, codebook_size=64),
        training=TrainingConfig(batch_size=8, num_epochs=3, save_every=1),
    )

def best_8d_config() -> OmniConfig:
    """
    Best production config for H-VQ-VAE with 8D OctupleMIDI tokens.
    Tuned for RTX 4060 (8GB VRAM):
      - hidden_dim=256, codebook_size=512 → target perplexity > 80
      - batch_size=64 fits in ~2GB VRAM, leaving room for Adapter + LLM
      - seq_len=128 ≈ 4 bars of 32nd-note grid (dense enough for phrasal structure)
      - CosineAnnealingLR over 20 epochs, saving every 5
    """
    return OmniConfig(
        data=DataConfig(
            max_files=2000,            # ~2000 LMD files for fast iteration
            seq_len=128,
            token_mode="octuple_8d",
            h5_cache="data/lmd_8d_best_cache.h5",
        ),
        model=ModelConfig(
            input_dim=-1,              # auto-resolved to 8
            hidden_dim=256,
            codebook_size=512,
            commitment_cost=1.0,
            variant="hierarchical",
        ),
        training=TrainingConfig(
            batch_size=64,
            num_epochs=20,
            learning_rate=5e-4,
            weight_decay=1e-4,
            grad_clip=1.0,
            num_workers=4,
            save_every=5,
            checkpoint_dir="checkpoints/best",
            device="auto",
        ),
    )


# ────────────────────────────────────────────────────────────────
# ExperimentConfig — toggle multiple variants for comparison (O/D)
# ────────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """
    Groups several OmniConfigs under one experiment umbrella.
    Each 'run' is an OmniConfig with a different model.variant.
    The shared DataConfig and TrainingConfig guarantee fair comparison.
    """
    name: str = "vqvae_comparison"
    runs: list[OmniConfig] = field(default_factory=list)

    @staticmethod
    def compare_smoke() -> "ExperimentConfig":
        """Side-by-side smoke test: hierarchical vs vanilla (both 5D)."""
        shared_data = DataConfig(max_files=100, seq_len=64, token_mode="miditok_5d")
        shared_train = TrainingConfig(batch_size=8, num_epochs=3, save_every=3)

        return ExperimentConfig(
            name="smoke_compare",
            runs=[
                OmniConfig(
                    data=shared_data,
                    model=ModelConfig(hidden_dim=64, codebook_size=64, variant="hierarchical"),
                    training=shared_train,
                ),
                OmniConfig(
                    data=shared_data,
                    model=ModelConfig(hidden_dim=64, codebook_size=64, variant="vanilla"),
                    training=shared_train,
                ),
            ],
        )

    @staticmethod
    def compare_tokens() -> "ExperimentConfig":
        """
        Compare 5-dim miditok tokens vs 8-dim OctupleMIDI tokens.
        Same model architecture and training config for fair comparison.
        Note: Different H5 caches because the feature dimensions differ.
              ModelConfig uses input_dim=-1 so it auto-resolves per run.
        """
        shared_train = TrainingConfig(batch_size=8, num_epochs=3, save_every=3)

        return ExperimentConfig(
            name="token_mode_compare",
            runs=[
                OmniConfig(
                    data=DataConfig(max_files=100, seq_len=64, token_mode="miditok_5d",
                                    h5_cache="data/lmd_5d_cache.h5"),
                    # input_dim=-1 → auto-resolved to 5 by resolve_input_dim()
                    model=ModelConfig(hidden_dim=64, codebook_size=64, variant="hierarchical"),
                    training=shared_train,
                ),
                OmniConfig(
                    data=DataConfig(max_files=100, seq_len=64, token_mode="octuple_8d",
                                    h5_cache="data/lmd_8d_cache.h5"),
                    # input_dim=-1 → auto-resolved to 8 by resolve_input_dim()
                    model=ModelConfig(hidden_dim=64, codebook_size=64, variant="hierarchical"),
                    training=shared_train,
                ),
            ],
        )

    @staticmethod
    def compare_full() -> "ExperimentConfig":
        """Full LMD run: hierarchical vs vanilla."""
        shared_data = DataConfig(max_files=None)
        shared_train = TrainingConfig(num_epochs=20)

        return ExperimentConfig(
            name="full_compare",
            runs=[
                OmniConfig(
                    data=shared_data,
                    model=ModelConfig(variant="hierarchical"),
                    training=shared_train,
                ),
                OmniConfig(
                    data=shared_data,
                    model=ModelConfig(variant="vanilla"),
                    training=shared_train,
                ),
            ],
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "runs": [r.to_dict() for r in self.runs]}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ ExperimentConfig saved → {path}")
