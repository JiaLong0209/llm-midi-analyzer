#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║   OmniLLM-Muse  —  All Commands Reference                       ║
# ║   Run from: /home/jialong/Programming/TKU_Assignments/114-2/NLP ║
# ╚══════════════════════════════════════════════════════════════════╝
# NOTE: Each section below is a standalone command.
#       Copy & paste the one you need into the terminal.
#       DO NOT run this file as a whole script.

set -e
cd "$(dirname "$0")/llm-midi-analyzer"

 
# ================== FULL PIPELINE V2 ============================

# Step 1: Align MidiCaps captions with LMD matched MIDI files
# poetry run python src/data_loader.py

# Step 2: Parallel Pre-tokenize LMD Dataset using MD5 filenames
# poetry run python src/preprocess_midi.py --input data/lmd_matched --output data/tokenized_8d
# 這一步會跑比較久（取決於 CPU 核心數，大約 10-20 分鐘）
poetry run python src/preprocess_midi.py \
    --input data/lmd_matched \
    --output data/tokenized_8d


# Step 3: Train Hierarchical 8D VQ-VAE (Stage 2)
poetry run python src/trainer/vqvae_trainer.py --mode best8d --data_dir data/tokenized_8d --epochs 30 --model_name omni_v4 --kmeans

# Step 4: Train Adapter + LoRA (Stage 3: FULL PRODUCTION ALIGNMENT)
#   - vqvae mode: Cross-Attention using Frozen VQ-VAE + Llama 3.2 1B
#   - seq_len 512: Captures ~16-32 bars of context for style/tempo detection

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py --mode vqvae \
                    --data_dir data/tokenized_8d \
                    --llm models/MIDI-LLM \
                    --vqvae checkpoints/best/omni_v4_epoch10.pt \
                    --seq_len 1024 \
                    --epochs 50 \
                    --max_files 100000 \
                    --batch_size 2

# Step 5: End-to-End Inference (Analyze raw MIDI)
# poetry run python src/generate_analysis.py --midi midi/251103_9.mid --checkpoint checkpoints/adapter/vqvae_epoch40.pt --seq_len 512

# ================================================================
# ================================================================
# ================================================================


# # 一鍵全自動掛機指令
# # poetry run python src/preprocess_midi.py --input data/lmd_matched --output data/tokenized_8d && \
# poetry run python src/trainer/vqvae_trainer.py --mode best8d --data_dir data/tokenized_8d --epochs 20 --model_name omni_v5 --kmeans && \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# poetry run python src/train_adapter.py --mode vqvae \
#                     --data_dir data/tokenized_8d \
#                     --llm models/MIDI-LLM \
#                     --vqvae checkpoints/best/omni_v5_epoch20.pt \
#                     --seq_len 1024 \
#                     --epochs 20 \
#                     --max_files 2000 \
#                     --batch_size 2


# Step 2: 訓練穩定版 VQ-VAE (omni_v5)
# 使用 5e-4 LR + 相對小節線，訓練 30 Epoch 確保 Codebook 充分收斂
poetry run python src/trainer/vqvae_trainer.py \
    --mode best8d \
    --data_dir data/tokenized_8d \
    --epochs 10 \
    --model_name omni_v6 \
    --kmeans && \

# Step 3: 訓練 Adapter + LoRA (生產級對齊)
# - vqvae 模式: 使用剛剛練好的 omni_v5 第 30 Epoch 權重
# - seq_len 1024: 捕捉約 64-128 顆音符，足夠判斷曲風與節奏
# - epochs 50: 針對 2000 個樣本進行深度對齊


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
    --epochs 5 \
    --lr 2e-4 \
    --output_dir checkpoints/adapter/exp_0504_1b_lora_epoch5


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
    --mode vqvae \
    --data_dir data/tokenized_8d \
    --llm models/MIDI-LLM \
    --vqvae checkpoints/best/omni_v6_epoch10.pt \
    --d_vq 256 \
    --seq_len 1024 \
    --max_files 2000 \
    --unsloth \
    --batch_size 2 \
    --epochs 5 \
    --lr 2e-4 \
    --output_dir checkpoints/adapter/exp_0504_unsloth_1b_lora_epoch5


# Step 4: 訓練 Adapter + LoRA (Full Set)
# 這次使用全部 20,000 個樣本，目標是讓模型學會長序列的完整風格與結構
# epochs 設為 5，避免 overfitting，並利用 L2 正規化保持泛化能力
# LoRA r=16, alpha=16: 捕捉更細緻的風格特徵（比 r=8 更高容量）

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
    --mode vqvae \
    --data_dir data/tokenized_8d \
    --llm models/MIDI-LLM \
    --vqvae checkpoints/best/omni_v6_epoch10.pt \
    --d_vq 256 \
    --seq_len 1024 \
    --max_files 20000 \
    --unsloth \
    --batch_size 4 \
    --epochs 5 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --output_dir checkpoints/adapter/exp_0504_unsloth_1b_lora_full

# ================================================================
# ================================================================
# ================================================================

# ── 1. DATA UTILITIES ───────────────────────────────────────────────
# 1a. Check dataset integrity (scans lmd_matched for corrupt MIDIs)
poetry run python data/check.py

# 1b. Pre-tokenize MIDI files → .npy files for fast training starts
#   --input     Source directory with raw .mid / .midi files
#   --output    Destination directory for .npy token arrays
#   --max_files Cap on how many files to process (omit for all)
#   --workers   Parallel CPU workers (default: CPU core count)
poetry run python src/preprocess_midi.py \
  --input data/lmd_matched \
  --output data/tokenized_8d \
  --max_files 50000


# ── 2. VQ-VAE SMOKE TESTS ───────────────────────────────────────────
# 2a. Quick sanity check (5-dim miditok, 100 files, 3 epochs)
poetry run python src/trainer/vqvae_trainer.py --mode smoke

# 2b. Quick sanity check (8-dim OctupleMIDI, 100 files, 3 epochs)
poetry run python src/trainer/vqvae_trainer.py --mode smoke8d


# ── 3. VQ-VAE CUSTOM TRAINING ───────────────────────────────────────
# Train from scratch with the pre-tokenized folder (fast startup)
#   --mode       Config preset: best8d (recommended for 8D OctupleMIDI)
#   --data_dir   Load from pre-tokenized .npy folder (skips MIDI extraction)
#   --hidden_dim GRU hidden size — must match checkpoint (default: 512)
#   --epochs     Total cumulative epochs (start_epoch + this)
#   --model_name Prefix for saved checkpoint filenames
#   --resume     Path to .pt checkpoint to continue from
#   --kmeans     Run K-means codebook initialization before first epoch
#   --force      Rebuild HDF5 cache even if it already exists
poetry run python src/trainer/vqvae_trainer.py --mode best8d \
  --data_dir data/tokenized_8d \
  --hidden_dim 512 \
  --epochs 100 \
  --model_name omni_v1 \
  --kmeans

# Resume from a previous checkpoint
poetry run python src/trainer/vqvae_trainer.py --mode best8d \
  --data_dir data/tokenized_8d \
  --hidden_dim 512 \
  --epochs 100 \
  --model_name omni_v1 \
  --resume checkpoints/best/hierarchical_epoch10.pt \
  --kmeans


# ── 4. VQ-VAE FULL TRAINING ─────────────────────────────────────────
# Full LMD dataset (miditok_5d token mode, 20 epochs)
poetry run python src/trainer/vqvae_trainer.py --mode full


# ── 5. VQ-VAE INSPECTION ────────────────────────────────────────────
# Inspect a checkpoint: generates MIDI pairs, codebook report & chart
#   --checkpoint  Path to trained .pt file
#   --data-cache  HDF5 cache built during training
#   --num-samples Number of samples to evaluate MSE on
#   --num-midi    Number of MIDI reconstruction pairs to save
poetry run python src/inspect_vqvae.py \
  --checkpoint checkpoints/best/omni_v1_epoch40.pt \
  --data-cache data/lmd_8d_best_cache.h5 \
  --num-samples 500 \
  --num-midi 20


# ── 6. ABLATION STUDY: MIDI-to-LLM ADAPTER ──────────────────────────
# Two paths for projecting OctupleMIDI into Llama 3.2 1B embedding space.
# Both paths output (B, N/4, 2048) for a fair token-length comparison.
# Log → output/adapter_log.jsonl  |  VRAM budget: RTX 4060 8GB
#
# Common args:
#   --mode       "direct" (MLP baseline) | "vqvae" (Cross-Attention proposed)
#   --data_dir   Directory of pre-tokenized .npy files
#   --llm        Path to the local MIDI-LLM (Llama 3.2 1B) model directory
#   --vqvae      Path to VQ-VAE checkpoint (required for vqvae mode only)
#   --d_vq       VQ-VAE hidden dim — must match the checkpoint (default: 512)
#   --seq_len    Input sequence length per sample (default: 128)
#   --batch_size Keep at 1 for 8GB VRAM; increase if GPU is shared
#   --max_files  Subset of .npy files to load (omit for full dataset)
#   --epochs     Number of training epochs
#   --lr         Learning rate (default: 2e-4)

# 6a. Path A — DirectMLPAdapter (baseline)
#     Architecture: Conv1d(stride=4) → ResidualMLP × 3 → SinusoidalPE → Linear(2048)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
  --mode direct \
  --data_dir data/tokenized_8d \
  --llm models/MIDI-LLM \
  --max_files 5000 \
  --batch_size 1 \
  --epochs 3

# 6b. Path B — CrossAttentionAdapter (proposed VQ-VAE path)
#     Architecture: FrozenVQVAE enc → GatedCrossAttn(Q=m4, K/V=m1) → SinusoidalPE → Linear(2048)
#     Gate formula: fused = tanh(α) × CrossAttn + m4  [Flamingo-style]
#     Codebook health check is enforced on startup (usage < 10% → error)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/train_adapter.py \
  --mode vqvae \
  --data_dir data/tokenized_8d \
  --llm models/MIDI-LLM \
  --vqvae checkpoints/best/omni_v1_epoch40.pt \
  --d_vq 512 \
  --max_files 5000 \
  --batch_size 1 \
  --epochs 3


# ── 7. EXPERIMENTS / COMPARISONS ────────────────────────────────────
# Compare 5D tokens vs 8D tokens on a smoke-scale dataset
poetry run python src/compare.py --mode tokens

# Compare Hierarchical vs Vanilla VQ-VAE (commented out by default)
# poetry run python src/compare.py --mode smoke

# ── 8. END-TO-END INFERENCE ────────────────────────────────────────
# Generate analysis for a raw MIDI file using the trained adapter.
#   --midi      Path to input .mid file
#   --checkpoint Path to adapter checkpoint (must contain LoRA weights)

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
poetry run python src/generate_analysis.py \
  --midi midi/no.16_260318_v6.mid \
  --checkpoint checkpoints/adapter/vqvae_epoch15.pt \
  --vqvae checkpoints/best/omni_v4_epoch02.pt \
  --seq_len 1024