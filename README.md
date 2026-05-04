# OmniLLM-Muse 🎵

A music intelligence system that combines **Hierarchical VQ-VAE tokenization** with **QLoRA fine-tuning** of Llama 3.2 1B to understand and analyze MIDI music semantically.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Pipeline](#2-full-pipeline)
3. [Stage 1 — MIDI Preprocessing](#3-stage-1--midi-preprocessing)
4. [Stage 2 — Hierarchical VQ-VAE Training](#4-stage-2--hierarchical-vq-vae-training)
5. [Stage 3 — MIDI-to-LLM Adapter Training (Ablation Study)](#5-stage-3--midi-to-llm-adapter-training)
6. [Installation](#6-installation)
7. [Quick Start](#7-quick-start)
8. [Configuration Reference](#8-configuration-reference)
9. [Project Structure](#9-project-structure)

---

## 1. Project Overview

OmniLLM-Muse is designed around a core hypothesis:

> **Raw MIDI tokens are too fine-grained for LLMs to reason about music structure. We need a semantic compression layer first.**

The system compresses raw MIDI sequences into hierarchical musical motifs (1-Bar and 4-Bar patterns) using a VQ-VAE, then projects those motifs into the embedding space of an LLM via a learned adapter. Two projection approaches are compared in an ablation study: a direct MLP baseline and a Cross-Attention architecture using the VQ-VAE representations.

---

## 2. Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OmniLLM-Muse Pipeline                           │
└─────────────────────────────────────────────────────────────────────────┘

 RAW MIDI FILES (.mid)
        │
        ▼
┌───────────────────────┐
│  Stage 1: Preprocess  │  28-core parallel extraction
│  OctupleMIDI (8D)     │  → Each note → 8 integers:
│  preprocess_midi.py   │    (Bar, Position, Program, Pitch,
└──────────┬────────────┘     Duration, Velocity, TimeSig, Tempo)
           │
           │  data/tokenized_8d/
           │  000001.npy, 000002.npy, …
           │  shape: (T, 8)  where T = num tokens
           │
           ▼
┌───────────────────────────────────────────────────────────────────────┐
│                  Stage 2: Hierarchical VQ-VAE Training                │
│                  vqvae_trainer.py + vqvae.py                          │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
   Input: (B, N, 8)  [normalized to -1 ~ 1]
        │
        ▼
┌──────────────────────┐
│  ResidualGRU Encoder │  2-layer Bidirectional GRU
│  enc_m1              │  + LayerNorm + Residual Shortcut
└──────────┬───────────┘
           │  (B, N, hidden_dim)  — 1-Bar feature map
           │
           ├──────────────────────────────────┐
           ▼                                  │
┌─────────────────────────┐         ┌─────────┴────────────────┐
│   VQ Layer (m1)         │         │  Conv1d (stride=4) + LN  │
│   EMA + Random Restart  │         │  enc_m4_conv             │
│   codebook_size entries │         └───────────┬──────────────┘
│                         │                     │
│   Nearest neighbor      │              (B, N/4, hidden_dim) — 4-Bar
│   lookup in codebook    │                     │
│                         │                     ▼
│   Straight-through      │         ┌───────────────────────────┐
│   gradient estimator    │         │   VQ Layer (m4)           │
│                         │         │   EMA + Random Restart    │
│   → idx_m1 (B, N)       │         │   → idx_m4 (B, N/4)       │
└──────────┬──────────────┘         └───────────────────────────┘
           │  q_m1 (B, N, hidden_dim)
           │
           ▼
┌──────────────────────┐
│  ResidualGRU Decoder │  2-layer Bidirectional GRU
│  dec                 │  + LayerNorm + Residual Shortcut
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output Head         │  Linear(hidden_dim → 8)
│  out_head            │  + Denormalize × 128
└──────────┬───────────┘
           │  Reconstructed OctupleMIDI (B, N, 8)
           │
           ▼
   Loss = β·commitment_m1 + β·commitment_m4 + 5.0·recon_MSE

        Saved: checkpoints/best/omni_v1_epochXX.pt
        │
        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  Stage 3: MIDI-to-LLM Adapter Training                │
│                  train_adapter.py + adapters.py                        │
│                  ── Ablation Study: Path A vs Path B ──                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Stage 1 — MIDI Preprocessing

### OctupleMIDI (8D) Format

Each MIDI note is converted into an 8-dimensional integer vector:

| Dim | Field    | Description                          | Range   |
|:---:|:---------|:-------------------------------------|:--------|
| 0   | Bar      | Which bar (measure) this note is in  | 0–127   |
| 1   | Position | Quantized beat position within bar   | 0–31    |
| 2   | Program  | MIDI instrument program              | 0–127   |
| 3   | Pitch    | MIDI note pitch (C4 = 60)            | 0–127   |
| 4   | Duration | Note length (quantized steps)        | 0–63    |
| 5   | Velocity | Key press strength (loudness)        | 0–31    |
| 6   | TimeSig  | Time signature index                 | 0–15    |
| 7   | Tempo    | Tempo bucket index                   | 0–48    |

### Numerical Examples

To help visualize how MIDI events are mapped to these 8 dimensions:

*   **A Middle C (C4) on Piano at Bar 2, Beat 1**: `[2, 0, 0, 60, 32, 20, 0, 24]`
    *   `Bar: 2` (The measure index)
    *   `Position: 0` (First 32nd note of the bar)
    *   `Program: 0` (Acoustic Grand Piano)
    *   `Pitch: 60` (Middle C)
    *   `Duration: 32` (A quarter note if bar division is 32)
    *   `Velocity: 20` (Moderate loudness ~80 in standard MIDI)
    *   `TimeSig: 0` (Default 4/4)
    *   `Tempo: 24` (Default 120 BPM)

*   **A Staccato Violin G4 at Bar 5, Last 8th Note**: `[5, 28, 40, 67, 4, 25, 0, 24]`
    *   `Position: 28` (Last 8th note in a 32-position grid)
    *   `Program: 40` (Violin)
    *   `Pitch: 67` (G4)
    *   `Duration: 4` (Very short/staccato)
    *   `Velocity: 25` (Forzando/Loud)

Tokenizing 50,000 MIDI files sequentially takes ~90 minutes. By pre-extracting once to `.npy` files, all subsequent training runs start in **seconds** with zero re-extraction cost.

```
data/lmd_matched/           data/tokenized_8d/
├─ genre1/                  ├─ 000001.npy   (T, 8)
│  ├─ track1.mid   ──→     ├─ 000002.npy
│  └─ track2.mid   ──→     ├─ 000003.npy
└─ genre2/          28      └─ …
   └─ track3.mid  cores     49,362 files extracted
```

**Command:**
```bash
poetry run python src/preprocess_midi.py \
  --input data/lmd_matched \
  --output data/tokenized_8d \
  --max_files 50000
```

---

## 4. Stage 2 — Hierarchical VQ-VAE Training

### Architecture Deep-Dive

#### ResidualGRU

```
Input (B, N, D_in)
      │
      ├──────── shortcut ──────────────────────────────┐
      │         Linear(D_in → D_out)                   │
      ▼                                                 │
 BiGRU (2 layers)                                       │
 hidden_dim × 2 (bidirectional concat)                  │
      │                                                 │
 Linear(hidden_dim×2 → D_out)                           │
      │                                                 │
      └──────── + ◄──────────────────────────────────┘
                │
           LayerNorm
                │
         Output (B, N, D_out)
```

Why residual? The 8 → hidden_dim dimensional jump is large. Residual connections prevent gradient vanishing deep in the 4-layer stack.

#### VectorQuantizer (with EMA + Random Restart)

```
Input (B, N, D)
      │
  LayerNorm         ← stabilizes distribution on unit sphere
      │
  Compute L2 distance to all K codebook entries
      │
  Select nearest entry → discrete code index
      │
      ├── EMA update (decay=0.99):
      │     cluster_size ← 0.99 × cluster_size + 0.01 × batch_counts
      │     codebook entry ← EMA average of assigned inputs
      │     (No gradient flows through codebook — stable training)
      │
      ├── Random Restart:
      │     If any codebook entry has 0 assignments this batch →
      │     reinitialize it from a random input in the batch
      │     (prevents "dead codes" / codebook collapse)
      │
  Straight-Through Estimator:
      quantized = input + (codebook_entry - input).detach()
      (gradient flows through as if quantization didn't happen)
      │
  Output: quantized (B, N, D), commitment_loss, indices (B, N)
```

#### K-Means Initialization

Before epoch 1, the trainer samples up to 2,000 sequences and runs `MiniBatchKMeans` on the flattened feature space. The resulting cluster centers seed the codebook.

```
Training data (K samples)
         │
   Extract m1 features  →  flatten  →  MiniBatchKMeans(n_clusters=codebook_size)
   Extract m4 features  →  flatten  →  MiniBatchKMeans(n_clusters=codebook_size)
         │
   Copy cluster centers into vq_m1.embedding + vq_m4.embedding
         │
   Perplexity at Epoch 0 is already > 1.0  (no collapse from scratch)
```

#### Training Loss

```
total_loss = 0.5 × commitment_loss_m1
           + 0.5 × commitment_loss_m4
           + 5.0 × reconstruction_MSE
```

The 5× weight on reconstruction prioritizes fidelity over codebook compaction in early training.

### Training Commands

```bash
# Fresh training from pre-tokenized data
poetry run python src/trainer/vqvae_trainer.py --mode best8d \
  --data_dir data/tokenized_8d \
  --hidden_dim 512 \
  --epochs 100 \
  --model_name omni_v1 \
  --kmeans

# Resume from existing checkpoint
poetry run python src/trainer/vqvae_trainer.py --mode best8d \
  --data_dir data/tokenized_8d \
  --hidden_dim 512 \
  --epochs 100 \
  --model_name omni_v1 \
  --resume checkpoints/best/omni_v1_epoch40.pt
```

### Inspecting Results

```bash
poetry run python src/inspect_vqvae.py \
  --checkpoint checkpoints/best/omni_v1_epoch40.pt \
  --data-cache data/lmd_8d_best_cache.h5 \
  --num-samples 500 \
  --num-midi 20
```

Output in `output/vqvae/`:
- `report.txt` — Perplexity, active code %, reconstruction MSE
- `codebook_usage.png` — Bar chart of codebook index frequency
- `midi/sample_XX_orig.mid` — Original MIDI
- `midi/sample_XX_recon.mid` — VQ-VAE reconstruction (listen to assess quality)

**Target metrics for production:**
| Metric | Minimum | Target |
|:---|:---:|:---:|
| m1 Perplexity | > 10.0 | > 50.0 |
| Codebook Usage | > 50% | > 90% |
| Reconstruction MSE | < 200 | < 100 |

---

## 5. Stage 3 — MIDI-to-LLM Adapter Training

### Overview: Two Paths, One Framework

Both adapters receive the same `.npy` data and output the same shape `(B, N/4, 2048)` before being prepended to Llama 3.2 1B's token sequence. The only difference is **how** they compress the MIDI.

### Path A — DirectMLPAdapter (Baseline)

```
OctupleMIDI (B, N, 8)   [normalized to -1 ~ 1]
      │
 Conv1d(8 → 2048, stride=4)    ← Preserves local time structure
 GELU                           ← (NOT mean-pooling — that destroys counterpoint)
      │
  (B, N/4, 2048)
      │
 ResidualBlock × 3:
   LayerNorm → Linear(2048 → 8192) → GELU → Linear(8192 → 2048) → + residual
      │
  (B, N/4, 2048)
      │
 SinusoidalPE                   ← Inject sequence position info
      │
 LayerNorm
      │
 Output (B, N/4, 2048)
```

### Path B — CrossAttentionAdapter with Frozen VQ-VAE (Proposed)

```
OctupleMIDI (B, N, 8)
      │
      ▼
┌─────────────────────────────┐
│  Frozen H-VQ-VAE Encoder    │   (loaded from checkpoints/best/omni_v1_*.pt)
│                             │
│  enc_m1 → q_m1 (B, N, D)   │   ← 1-Bar tokens: fine-grained note events
│  enc_m4_conv → q_m4 (B, N/4, D)  ← 4-Bar tokens: phrase-level motifs
└──────────────┬──────────────┘
               │
               │   Q = m4 tokens  (B, N/4, D)  — phrase queries
               │   K = m1 tokens  (B, N,   D)  — note-level context
               │   V = m1 tokens  (B, N,   D)
               │
        MultiheadAttention(Q, K, V)   [8 heads]
               │
               │   attn_out (B, N/4, D)
               │
        Flamingo-style Gate:
          fused = m4 + tanh(α) × attn_out
          (α = nn.Parameter(0) at init → starts as pure m4 path,
           gradually learns to blend in fine-grained 1-Bar context)
               │
        SinusoidalPE
               │
        LayerNorm → Linear(D → 2048)
               │
        Output (B, N/4, 2048)
```

### Why Gated Cross-Attention?

The 4-Bar tokens capture phrase structure (the "what") while 1-Bar tokens capture note-level detail (the "how"). Cross-Attention lets the model decide how much fine-grained context to absorb per phrase. The `tanh(α)` gate ensures training starts stable (gate=0 means pure 4-Bar path) and only opens as the adapter gains confidence.

### LLM Integration

```
Adapter Output    +    Text Prompt Embeddings
(B, N/4, 2048)         (B, L, 2048)
      │                        │
      └────────── cat ─────────┘
                  │
           (B, N/4+L, 2048)
                  │
       Llama 3.2 1B (4-bit QLoRA)
         trainable: ~0.2% parameters
         (q_proj, k_proj, v_proj, o_proj via LoRA r=16)
                  │
            Cross-Entropy Loss
            on text token predictions only
            (prefix N/4 tokens masked with label=-100)
```

### Loss Calculation Flow

The model uses **Standard Cross-Entropy Loss** to optimize the adapter. The key is in how we handle the multi-modal sequence:

1.  **Forward Pass**: The system concatenates the `Music_Prefix` and `Text_Embeddings`. The LLM processes the sequence as a single stream of 2048-dim vectors.
2.  **Label Masking**:
    *   **Music Indices**: All labels for the MIDI prefix tokens are set to `-100` (the "ignore index" in PyTorch).
    *   **Text Indices**: Labels are set to the actual text token IDs generated by the tokenizer.
3.  **Cross-Entropy Evaluation**:
    *   The loss is calculated **only on the text part**.
    *   Formula: $L = -\frac{1}{M} \sum \log P(\text{Text}_i \mid \text{Music\_Prefix}, \text{Context}_{<i})$
4.  **Optimization**: Gradients flow back through the LLM (via LoRA adapters) into the `Adapter` weights. This forces the adapter to project MIDI into a "language" that the LLM understands as a valid prefix for the musical analysis task.

### Why Path B Wins

The ablation study in `adapter_log.jsonl` typically shows Path B starting with **~60% lower loss** than Path A. This confirms that the Hierarchical VQ-VAE codebook provides a "semantic vocabulary" that is significantly closer to language than raw OctupleMIDI numbers.

### Training Commands

```bash
# Path A — Direct MLP baseline
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
  --mode direct \
  --data_dir data/tokenized_8d \
  --llm models/MIDI-LLM \
  --max_files 5000 --batch_size 1 --epochs 3

# Path B — VQ-VAE Cross-Attention (requires trained VQ-VAE checkpoint)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
  --mode vqvae \
  --data_dir data/tokenized_8d \
  --llm models/MIDI-LLM \
  --vqvae checkpoints/best/omni_v1_epoch40.pt \
  --d_vq 512 \
  --max_files 5000 --batch_size 1 --epochs 3
```

Comparison log → `output/adapter_log.jsonl`:
```json
{"epoch": 1, "mode": "direct", "avg_loss": 0.1764, "vram_mb": 2838, "token_length_out": 16}
{"epoch": 1, "mode": "vqvae",  "avg_loss": 0.1102, "vram_mb": 3104, "token_length_out": 16}
```

---

## 6. Installation

### Prerequisites
- **Hardware**: NVIDIA GPU with **8GB+ VRAM** (e.g., RTX 3060/4060) is recommended for training.
- **Python**: 3.11+ (managed via `pyenv` or `conda`).
- **Dependency Manager**: [Poetry](https://python-poetry.org/) 2.0+.

### Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd llm-midi-analyzer

# Install exact Python version and dependencies
pyenv install 3.11.9
poetry env use 3.11.9
poetry install --no-root

# Verify CUDA availability
poetry run python -c "import torch; print(f'Using GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'CUDA NOT FOUND')"
```

### Model & Data Preparation
1. **LLM Base**: Place/link Llama 3.2 1B (unsloth optimized version recommended) into `models/MIDI-LLM/`.
2. **Dataset**: Place your raw MIDI files into `data/lmd_matched/`.

---

## 7. Quick Start

Follow these steps to run the full pipeline from scratch:

### Step 1: Pre-tokenize MIDI
Extract OctupleMIDI (8D) features into serialized `.npy` files for high-speed training.
```bash
poetry run python src/preprocess_midi.py \
  --input data/lmd_matched \
  --output data/tokenized_8d \
  --max_files 20000
```

### Step 2: Train Hierarchical VQ-VAE
Learn the musical "vocabulary" (1-Bar and 4-Bar latent codes).
```bash
poetry run python src/trainer/vqvae_trainer.py --mode best8d \
  --data_dir data/tokenized_8d \
  --hidden_dim 512 \
  --epochs 100 \
  --kmeans
```

### Step 3: Verify Reconstruction
Ensure the VQ-VAE can accurately "listen" and reconstruct MIDI.
```bash
poetry run python src/inspect_vqvae.py \
  --checkpoint checkpoints/best/omni_v1_epoch100.pt \
  --data-cache data/lmd_8d_best_cache.h5
```

### Step 4: Train MIDI-to-LLM Adapter
Fine-tune the projection layer and Llama LoRA to analyze the latent codes.
```bash
# Note: expandable_segments helps prevent VRAM fragmentation on 8GB cards
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
  --mode vqvae \
  --data_dir data/tokenized_8d \
  --llm models/MIDI-LLM \
  --vqvae checkpoints/best/omni_v1_epoch100.pt \
  --epochs 5 \
  --batch_size 1
```

### Step 4: Alternative example (unsloth 1b qlora 8 epochs)
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
    --mode vqvae \
    --data_dir data/tokenized_8d \
    --llm models/MIDI-LLM \
    --vqvae checkpoints/best/omni_v6_epoch10.pt \
    --d_vq 256 \
    --seq_len 1024 \
    --max_files 2000 \
    --batch_size 2 \
    --epochs 8 \
    --lr 2e-4 \
    --output_dir checkpoints/adapter/exp_0504_1b_lora_epoch5
```

---

## 8. Configuration Reference

All hyper-parameters live in `src/config.py` as typed dataclasses.

| Config Class | Key Fields |
|:---|:---|
| `DataConfig` | `lmd_dir`, `h5_cache`, `seq_len`, `max_files`, `token_mode` |
| `ModelConfig` | `hidden_dim`, `codebook_size`, `commitment_cost`, `variant` |
| `TrainingConfig` | `batch_size`, `num_epochs`, `lr`, `grad_clip`, `save_every` |
| `AdapterConfig` | `projection_mode`, `d_llm`, `d_vq`, `vqvae_checkpoint`, `qlora_r` |

---

## 9. Project Structure

```
llm-midi-analyzer/
├─ src/
│  ├─ config.py                  # All config dataclasses (SOLID)
│  ├─ preprocess_midi.py         # Stage 1: parallel MIDI → .npy
│  ├─ inspect_vqvae.py           # VQ-VAE quality evaluation
│  ├─ train_adapter.py           # Stage 3: adapter ablation trainer
│  ├─ generate_analysis.py       # End-to-End Inference (MIDI → Text)
│  ├─ models/
│  │  ├─ vqvae.py               # ResidualGRU, VectorQuantizer, HierarchicalVQVAE
│  │  ├─ octuple.py             # OctupleMIDI 8D extractor + MIDI reconstruction
│  │  └─ adapters.py            # DirectMLPAdapter, CrossAttentionAdapter, Factory
│  └─ trainer/
│     └─ vqvae_trainer.py       # Stage 2: dataset, K-means init, training loop
├─ data/
│  ├─ lmd_matched/              # Raw LMD MIDI files (~17,000)
│  └─ tokenized_8d/             # Pre-tokenized .npy files (run preprocess once)
├─ checkpoints/
│  ├─ best/                     # Production VQ-VAE checkpoints
│  └─ adapter/                  # Adapter weights per mode
├─ output/
│  ├─ vqvae/                    # Inspection reports and MIDI pairs
│  └─ adapter_log.jsonl         # Ablation comparison metrics
├─ models/
│  └─ MIDI-LLM/                 # Llama 3.2 1B base model (local)
├─ MIDI-LLM/                    # MIDI-LLM generation scripts
└─ run.sh                       # All commands reference
```

---

## 10. Inference: Generate Analysis

Once an adapter is trained, you can use it to analyze raw `.midi` files. The system will handle the entire pipeline from extraction to LLM text generation.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/generate_analysis.py \
  --midi tests/sample.mid \
  --checkpoint checkpoints/adapter/vqvae_epoch03.pt
```

**Note**: Ensure your checkpoint file contains the **LoRA weights**. If you trained with an older version of the script, results will be random until you rerun the training to save a "full" checkpoint.