"""
adapters.py — MIDI-to-LLM Projection Adapters (Ablation Study)
===============================================================
Two paths, both output (B, N/4, D_llm) for fair comparison.

Path A — DirectMLPAdapter:
    Conv1d(stride=4) → Residual MLP → SinusoidalPE → Linear(d_llm)

Path B — CrossAttentionAdapter:
    Frozen H-VQ-VAE encoder → Gated Cross-Attention (Flamingo)
    gate = tanh(α) for bounded gradient → SinusoidalPE → Linear(d_llm)

Toggle: AdapterFactory.build(config)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Positional Encoding
# ──────────────────────────────────────────────────────────────────────
class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding up to max_len positions."""
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ──────────────────────────────────────────────────────────────────────
# Path A — Direct MLP Adapter
# ──────────────────────────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """MLP block with residual connection for stable large-dim jumps."""
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 4),
            nn.GELU(),
            nn.Linear(d * 4, d),
        )

    def forward(self, x):
        return x + self.net(x)


class DirectMLPAdapter(nn.Module):
    """
    Path A baseline adapter.
    Conv1d(stride=4) preserves local temporal structure instead of
    mean-pooling (which destroys counterpoint relationships).
    """
    def __init__(self, config):
        super().__init__()
        d = config.d_llm
        # Stride=4 downsamples N → N/4, matching Path B's output length
        self.conv = nn.Sequential(
            nn.Conv1d(config.input_dim, d, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
        )
        # 3-layer residual MLP to stabilize 8 → d_llm gradient flow
        self.mlp = nn.Sequential(
            ResidualBlock(d),
            ResidualBlock(d),
            ResidualBlock(d),
        )
        self.pe = SinusoidalPE(d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 8) OctupleMIDI, already normalized to [-1, 1]
        Returns:
            (B, N/4, D_llm)
        """
        # Conv expects (B, C, L): transpose
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, N/4, d)
        h = self.mlp(h)
        h = self.pe(h)
        return self.norm(h)


# ──────────────────────────────────────────────────────────────────────
# Path B — Gated Cross-Attention Adapter (Flamingo-style)
# ──────────────────────────────────────────────────────────────────────
class CrossAttentionAdapter(nn.Module):
    """
    Path B semantic adapter.

    Gate formula (Flamingo):
        fused = tanh(gate) * CrossAttn(Q=m4, K/V=m1) + m4

    tanh ensures gate stays bounded, preventing training explosion.
    gate is initialized to 0 → model starts from pure 4-Bar path,
    then gradually learns to blend 1-Bar fine-grained context.
    """
    def __init__(self, config):
        super().__init__()
        from models.vqvae import HierarchicalVQVAE

        self.d_vq = config.d_vq
        d = config.d_llm

        # Load frozen pre-trained VQ-VAE (encoder only)
        self.vqvae = HierarchicalVQVAE(
            input_dim=config.input_dim,
            hidden_dim=config.d_vq,
            codebook_size=config.codebook_size,
        )
        if config.vqvae_checkpoint:
            self._load_and_validate(config.vqvae_checkpoint, config.vqvae_min_codebook_usage)

        if config.freeze_vqvae:
            for p in self.vqvae.parameters():
                p.requires_grad_(False)

        # Gated Cross-Attention: Q=m4, K/V=m1
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_vq,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(self.d_vq)
        self.norm_kv = nn.LayerNorm(self.d_vq)
        # Flamingo gate — scalar initialized to 0
        self.gate = nn.Parameter(torch.zeros(1))

        # Project to LLM embedding dim
        self.pe = SinusoidalPE(self.d_vq)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.d_vq),
            nn.Linear(self.d_vq, d),
        )

    def _load_and_validate(self, checkpoint_path: str, min_usage: float = 0.10):
        """Load VQ-VAE weights and run a codebook health check."""
        import os
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  VQ-VAE checkpoint not found: {checkpoint_path}. Starting from scratch.")
            return

        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model_state", state)
        try:
            self.vqvae.load_state_dict(sd, strict=True)
            print(f"✅ VQ-VAE weights loaded from {checkpoint_path}")
        except RuntimeError as e:
            print(f"❌ VQ-VAE Loading Error: Dimension mismatch or corrupted checkpoint.")
            print(f"   Ensure --d_vq ({self.d_vq}) matches the checkpoint hidden_dim.")
            raise e

        # Health check on codebook usage via EMA counts
        cluster_size = sd.get("vq_m1.ema_cluster_size", None)
        if cluster_size is not None:
            total = cluster_size.sum().item()
            if total > 0:
                active = (cluster_size > 0.5).sum().item()
                usage = active / len(cluster_size)
                print(f"   📊 m1 Codebook usage: {active}/{len(cluster_size)} ({usage*100:.1f}%)")
                if usage < min_usage:
                    raise RuntimeError(
                        f"❌ Codebook collapse detected! Usage {usage*100:.1f}% < {min_usage*100:.0f}% threshold.\n"
                        f"   Please retrain the VQ-VAE with --kmeans flag before using the VQ-VAE adapter."
                    )

    def _encode(self, x: torch.Tensor):
        """Extract m1 (1-Bar) and m4 (4-Bar) features from the frozen encoder."""
        x_norm = x.float() / 128.0 - 1.0
        m1 = self.vqvae.enc_m1(x_norm)                         # (B, N, d_vq)
        m4 = F.elu(self.vqvae.enc_m4_conv(m1.transpose(1, 2))) \
               .transpose(1, 2)                                  # (B, N/4, d_vq)
        m4 = self.vqvae.enc_m4_ln(m4)
        return m1, m4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 8) OctupleMIDI, NOT yet normalized (done internally)
        Returns:
            (B, N/4, D_llm)
        """
        m1, m4 = self._encode(x)

        # Gated Cross-Attention: Q=m4 attends over K/V=m1
        q = self.norm_q(m4)
        kv = self.norm_kv(m1)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, N/4, d_vq)

        # Flamingo gate: tanh bounds gradient, gate=0 → purely Q path at init
        fused = m4 + torch.tanh(self.gate) * attn_out

        fused = self.pe(fused)
        return self.proj(fused)


# ──────────────────────────────────────────────────────────────────────
# Path C — MusicBERT / Pretrained Transformer Adapter
# ──────────────────────────────────────────────────────────────────────
class MusicBERTAdapter(nn.Module):
    """
    Path C — Pretrained Encoder Adapter.
    Uses a RoBERTa-based architecture (MusicBERT style) to extract
    deep musical semantic features.
    
    Downsamples N -> N/4 using a stride-4 convolution before BERT
    to maintain fair comparison with Path A and Path B.
    """
    def __init__(self, config):
        super().__init__()
        from transformers import RobertaConfig, RobertaModel
        
        d = config.d_llm
        
        # 1. Load Pretrained Encoder with robust prefix handling
        try:
            print(f"📦 Loading MusicBERT backbone from: {config.musicbert_model_path}")
            import torch
            from transformers import AutoModel
            
            # Resolve dtype from config string
            dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
            target_dtype = dtype_map.get(config.torch_dtype, torch.float32)
            print(f"   Using precision: {config.torch_dtype}")

            self.bert = AutoModel.from_pretrained(
                config.musicbert_model_path, 
                add_pooling_layer=False,
                torch_dtype=target_dtype
            )
            
            # Check if it loaded correctly, sometimes it loads as a wrapper
            if hasattr(self.bert, "roberta"):
                print("   Found .roberta attribute, extracting backbone...")
                self.bert = self.bert.roberta
            elif hasattr(self.bert, "bert"):
                print("   Found .bert attribute, extracting backbone...")
                self.bert = self.bert.bert
                
        except Exception as e:
            print(f"⚠️  Standard loading had issues: {e}. Attempting manual state_dict fix...")
            try:
                from transformers import RobertaConfig, RobertaModel
                import torch
                # Initialize model with config only
                bert_cfg = RobertaConfig.from_pretrained(config.musicbert_model_path)
                self.bert = RobertaModel(bert_cfg)
                
                # Load state dict manually to strip prefixes
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=config.musicbert_model_path, filename="pytorch_model.bin")
                sd = torch.load(model_file, map_location="cpu")
                
                # Strip 'bert.' or 'roberta.' prefixes
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("bert."): new_sd[k[5:]] = v
                    elif k.startswith("roberta."): new_sd[k[8:]] = v
                    else: new_sd[k] = v
                
                self.bert.load_state_dict(new_sd, strict=False)
                print("   ✅ Manual state_dict fix successful.")
            except Exception as e2:
                print(f"❌ All loading attempts failed ({e2}). Initializing random RoBERTa.")
                # Fallback configuration
                bert_cfg = RobertaConfig(
                    vocab_size=1, # We don't use the embedding layer tokens
                    hidden_size=768,
                    num_hidden_layers=6,
                    num_attention_heads=12,
                    intermediate_size=3072,
                )
                self.bert = RobertaModel(bert_cfg)

        # 2. Resolve hidden dimension dynamically
        self.d_model = self.bert.config.hidden_size
        print(f"   Detected hidden_size: {self.d_model}")

        # 3. Initialize layers with correct dimension
        # Input Projection & Downsampling (N -> N/4)
        self.input_proj = nn.Conv1d(config.input_dim, self.d_model, kernel_size=7, stride=4, padding=3)

        if config.freeze_vqvae: # Reusing this flag for encoder freezing
            print(f"❄️  Freezing MusicBERT encoder ({self.d_model})")
            for p in self.bert.parameters():
                p.requires_grad_(False)

        # Output Projection to LLM dim
        self.pe = SinusoidalPE(self.d_model)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 8) OctupleMIDI
        Returns:
            (B, N/4, D_llm)
        """
        # x: (B, N, 8) -> norm to [-1, 1]
        x_norm = x.float() / 128.0 - 1.0
        
        # Downsample & Project: (B, N, 8) -> (B, d_model, N/4) -> (B, N/4, d_model)
        h = self.input_proj(x_norm.transpose(1, 2)).transpose(1, 2)
        
        # BERT Encoding
        # We skip the standard embedding layer and use our projected features as inputs_embeds
        bert_out = self.bert(inputs_embeds=h)
        h = bert_out.last_hidden_state # (B, N/4, d_model)
        
        h = self.pe(h)
        return self.output_proj(h)


# ──────────────────────────────────────────────────────────────────────
# Factory (Factory Pattern)
# ──────────────────────────────────────────────────────────────────────
class AdapterFactory:
    @staticmethod
    def build(config) -> nn.Module:
        """Build and return the correct adapter based on config.projection_mode."""
        if config.projection_mode == "direct":
            print("🏗️  Building DirectMLPAdapter (Path A)")
            return DirectMLPAdapter(config)
        elif config.projection_mode == "vqvae":
            print("🏗️  Building CrossAttentionAdapter (Path B)")
            return CrossAttentionAdapter(config)
        elif config.projection_mode == "musicbert":
            print(f"🏗️  Building MusicBERTAdapter (Path C) using {config.musicbert_model_path}")
            return MusicBERTAdapter(config)
        else:
            raise ValueError(f"Unknown projection_mode: {config.projection_mode!r}")
