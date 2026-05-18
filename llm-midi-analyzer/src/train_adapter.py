"""
train_adapter.py — Unified MIDI-to-LLM Adapter Training Script
===============================================================
Supports both:
  - Path A: --mode direct  (DirectMLPAdapter)
  - Path B: --mode vqvae   (CrossAttentionAdapter)

Uses bitsandbytes 4-bit quantization + PEFT QLoRA.
VRAM budget target: RTX 4060 8GB.

Usage:
  python src/train_adapter.py --mode direct --max_files 100 --epochs 1
  python src/train_adapter.py --mode vqvae --epochs 1 \\
      --vqvae checkpoints/best/omni_v1_epoch40.pt
"""

import sys
import os
import json
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from config import AdapterConfig
from models.adapters import AdapterFactory


# ──────────────────────────────────────────────────────────────────────
# VRAM Utilities
# ──────────────────────────────────────────────────────────────────────
def vram_used_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def maybe_empty_cache(seq_len: int, threshold: int = 2048):
    """Free fragmented cache when sequence length risks OOM."""
    if torch.cuda.is_available() and seq_len > threshold:
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────
class NpyMIDIDataset(Dataset):
    """
    Loads pre-tokenized .npy OctupleMIDI files matched with MidiCaps descriptions.
    """
    def __init__(self, data_dir: str, jsonl_path: str = "data/mapped_midicaps.jsonl", seq_len: int = 128, max_files: int = None):
        self.seq_len = seq_len
        self.samples = []

        if not os.path.exists(jsonl_path):
            print(f"⚠️ Warning: {jsonl_path} not found. Attempting fallback raw .npy loading.")
            # Fallback for debugging when JSONL isn't ready
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
            if max_files:
                import random; random.shuffle(files)
                files = files[:max_files]
            
            for f in files:
                try:
                    arr = np.load(f).astype(np.float32)
                    for start in range(0, len(arr) - seq_len, seq_len // 2):
                        self.samples.append({"music": arr[start : start + seq_len], "caption": "An electronic track with synthesizers."})
                except Exception:
                    continue
        else:
            print(f"📂 Loading mappings from {jsonl_path}...")
            entries = []
            with open(jsonl_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        md5 = entry.get('location', '')
                        if not md5: continue
                        npy_path = os.path.join(data_dir, f"{md5}.npy")
                        if os.path.exists(npy_path):
                            entries.append(entry)
            
            if max_files:
                import random; random.shuffle(entries)
                entries = entries[:max_files]

            for entry in tqdm(entries, desc="Loading paired dataset", ncols=100):
                md5 = entry.get('location', '')
                caption = entry.get('caption', '')
                if not caption: continue
                npy_path = os.path.join(data_dir, f"{md5}.npy")
                
                try:
                    arr = np.load(npy_path).astype(np.float32)
                    for start in range(0, len(arr) - seq_len, seq_len // 2):
                        self.samples.append({"music": arr[start : start + seq_len], "caption": caption})
                except Exception:
                    continue

        print(f"📊 Total samples (captioned chunks): {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "music": torch.from_numpy(item["music"]),
            "caption": item["caption"]
        }


# ──────────────────────────────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────────────────────────────
def load_llm_with_qlora(config: AdapterConfig):
    """Load Llama 3.2 1B in 4-bit + QLoRA (supports Unsloth toggle)."""
    model_path = os.path.abspath(config.llm_model_path)
    
    if config.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            print(f"🚀 [Unsloth] Loading LLM from: {model_path}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=config.seq_len,
                load_in_4bit=True,
                device_map="auto",
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=config.qlora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=config.qlora_alpha,
                lora_dropout=0,  # Unsloth optimized with 0 dropout
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            return tokenizer, model
        except ImportError:
            print("⚠️ Unsloth not installed. Falling back to standard transformers.")

    print(f"🤖 [Standard] Loading LLM from: {model_path}")
    # Resolve compute_dtype from config
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map.get(config.torch_dtype, torch.float16) # Default to f16 for 4bit

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Enable gradient checkpointing to reduce VRAM peak
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required for grad checkpointing

    lora_config = LoraConfig(
        r=config.qlora_r,
        lora_alpha=config.qlora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return tokenizer, model


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────
def build_music_prefix(adapter, batch: torch.Tensor, device: str):
    """
    Run MIDI through adapter to get prefix embeddings for the LLM.
    Returns: (B, N/4, d_llm)
    """
    x = batch.to(device)
    maybe_empty_cache(x.size(1))
    return adapter(x)  # (B, N/4, d_llm)


def train(args):
    # Reduce VRAM fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Adapter config
    acfg = AdapterConfig(
        projection_mode=args.mode,
        d_vq=args.d_vq,
        vqvae_checkpoint=args.vqvae,
        llm_model_path=args.llm,
        seq_len=args.seq_len,
        use_unsloth=args.unsloth,
        qlora_r=args.lora_r,
        qlora_alpha=args.lora_alpha,
        musicbert_model_path=args.musicbert,
        torch_dtype=args.dtype,
    )

    # Build adapter
    adapter = AdapterFactory.build(acfg).to(device)
    adapter.train()

    # Dataset & Validation Split
    full_dataset = NpyMIDIDataset(args.data_dir, seq_len=args.seq_len, max_files=args.max_files)
    val_size = int(0.05 * len(full_dataset))
    val_size = max(1, val_size) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # LLM + QLoRA
    tokenizer, llm = load_llm_with_qlora(acfg)
    llm.train()

    # Paged AdamW uses CPU-side paging to reduce peak VRAM for optimizer states
    from bitsandbytes.optim import PagedAdamW8bit
    optimizer = PagedAdamW8bit(
        list(filter(lambda p: p.requires_grad, list(adapter.parameters()) + list(llm.parameters()))),
        lr=args.lr,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "adapter_log.jsonl")
    
    # TensorBoard Setup
    log_dir = os.path.join("runs", f"adapter_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"📊 TensorBoard logs → {log_dir}")

    print(f"\n{'='*60}")
    print(f"🚀 Training — Mode: {args.mode.upper()}")
    print(f"   Adapter params: {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}")
    print(f"{'='*60}\n")

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=110)

            for batch in pbar:
                # Extract music and captions
                music_batch = batch["music"].to(device)
                captions = batch["caption"]  # Tuple of strings

                optimizer.zero_grad()

                # 1. Get music prefix embedding from adapter
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    music_prefix = build_music_prefix(adapter, music_batch, device)  # (B, N/4, d_llm)

                # 2. Prompt embeddings (using real MidiCaps)
                # Instruct format encourages the model to act as a music analyzer
                prompts = [
                    f"{acfg.theory_context}\nAnalyze the musical structure of this piece. Description: {cap}" if acfg.theory_context else f"Analyze the musical structure of this piece. Description: {cap}"
                    for cap in captions
                ]

                tok = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                text_embeds = llm.get_input_embeddings()(tok.input_ids)  # (1, L, d)
                
                # Cast to LLM dtype (bf16) to prevent dtype mismatch
                llm_dtype = next(llm.parameters()).dtype
                music_prefix = music_prefix.to(llm_dtype)
                text_embeds = text_embeds.to(llm_dtype)

                # 3. Concatenate music prefix + text as LLM input
                full_embeds = torch.cat([music_prefix, text_embeds], dim=1)
                # Labels: -100 for prefix (ignore loss), actual text tokens for rest
                prefix_len = music_prefix.size(1)
                labels = torch.cat([
                    torch.full((music_prefix.size(0), prefix_len), -100, dtype=torch.long, device=device),
                    tok.input_ids
                ], dim=1)

                out = llm(inputs_embeds=full_embeds, labels=labels)
                loss = out.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(adapter.parameters()) + list(llm.parameters()), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "vram": f"{vram_used_mb():.0f}MB"})

            avg_train_loss = epoch_loss / len(loader) if len(loader) > 0 else 0.0
            
            # Validation Loop
            adapter.eval()
            llm.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    music_batch = batch["music"].to(device)
                    captions = batch["caption"]
                    
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        music_prefix = build_music_prefix(adapter, music_batch, device)
                    
                    prompts = [
                        f"{acfg.theory_context}\nAnalyze the musical structure of this piece. Description: {cap}" if acfg.theory_context else f"Analyze the musical structure of this piece. Description: {cap}"
                        for cap in captions
                    ]
                    tok = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
                    text_embeds = llm.get_input_embeddings()(tok.input_ids)
                    
                    llm_dtype = next(llm.parameters()).dtype
                    music_prefix = music_prefix.to(llm_dtype)
                    text_embeds = text_embeds.to(llm_dtype)
                    
                    full_embeds = torch.cat([music_prefix, text_embeds], dim=1)
                    prefix_len = music_prefix.size(1)
                    labels = torch.cat([
                        torch.full((music_prefix.size(0), prefix_len), -100, dtype=torch.long, device=device),
                        tok.input_ids
                    ], dim=1)
                    
                    out = llm(inputs_embeds=full_embeds, labels=labels)
                    val_loss_total += out.loss.item()
                    
            avg_val_loss = val_loss_total / len(val_loader) if len(val_loader) > 0 else 0.0
            adapter.train()
            llm.train()
            
            vram = vram_used_mb()
            seq_len_report = args.seq_len // 4  # after stride/vqvae compression

            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "mode": args.mode,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(avg_val_loss, 6),
                "vram_mb": round(vram, 1),
                "token_length_out": seq_len_report,
            }
            ts = log_entry["timestamp"]
            print(f"[{ts}] 📈 Epoch {epoch} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | vram={vram:.0f}MB")

            # TensorBoard Logging
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
            writer.add_scalar("Stats/VRAM_MB", vram, epoch)
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving emergency checkpoint...")
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = os.path.join(args.output_dir, f"{args.mode}_interrupted.pt")
        lora_state_dict = {k: v for k, v in llm.state_dict().items() if "lora_" in k}
        torch.save({
            "adapter": adapter.state_dict(),
            "lora": lora_state_dict,
            "config": acfg.to_dict(),
        }, ckpt)
        print(f"  💾 Emergency checkpoint saved → {ckpt}")
        return

    # Save adapter and LoRA weights
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt = os.path.join(args.output_dir, f"{args.mode}_epoch{args.epochs:02d}.pt")
    
    # Extract only the trainable LoRA parameters to keep checkpoint small
    lora_state_dict = {k: v for k, v in llm.state_dict().items() if "lora_" in k}
    
    torch.save({
        "adapter": adapter.state_dict(),
        "lora": lora_state_dict,
        "config": acfg.to_dict(),
    }, ckpt)
    print(f"\n💾 Full Model saved (Adapter + LoRA) → {ckpt}")
    print(f"📄 Log → {log_path}")
    print("\n✅ Training complete!")


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI-to-LLM Adapter Trainer (Ablation Study)")
    parser.add_argument("--mode", choices=["direct", "vqvae", "musicbert"], required=True)
    parser.add_argument("--data_dir", default="data/tokenized_8d", help="Pre-tokenized .npy directory")
    parser.add_argument("--llm", default="models/MIDI-LLM", help="LLM model path")
    parser.add_argument("--vqvae", type=str, default=None, help="Path to VQ-VAE checkpoint (vqvae mode only)")
    parser.add_argument("--musicbert", type=str, default="roberta-base", help="HF path or local dir for MusicBERT encoder")
    parser.add_argument("--d_vq", type=int, default=256, help="VQ-VAE hidden dim (must match checkpoint)")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2, help="Small batch for 8GB VRAM")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--output_dir", default="checkpoints/adapter", help="Directory to save logs and checkpoints")
    parser.add_argument("--unsloth", action="store_true", help="Enable Unsloth for 2x faster training")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16", help="Precision for models")
    args = parser.parse_args()
    train(args)
