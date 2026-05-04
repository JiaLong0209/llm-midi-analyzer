"""
inspect_vqvae.py — VQ-VAE Output Inspector
===========================================
Loads a trained checkpoint, runs samples through it, and saves:
  output/vqvae/
  ├── report.txt                 — human-readable summary
  ├── codebook_usage.png         — bar chart visualization
  ├── midi/
  │   ├── sample_00_orig.mid     — ground truth MIDI
  │   └── sample_00_recon.mid    — VQ-VAE reconstructed MIDI
  └── ... (numpy arrays)

Usage:
  poetry run python src/inspect_vqvae.py \
      --checkpoint checkpoints/best/hierarchical_epoch10.pt \
      --data-cache data/lmd_8d_best_cache.h5
"""
import sys, os, argparse
import numpy as np
import torch
import h5py
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from models.vqvae import HierarchicalVQVAE
from models.octuple import octuple8d_to_midi

OUT_DIR = "output/vqvae"
MIDI_OUT = os.path.join(OUT_DIR, "midi")


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def load_samples(h5_path: str, seq_len: int, n: int) -> torch.Tensor:
    """Pull the first N valid windows from the H5 cache."""
    samples = []
    with h5py.File(h5_path, "r") as f:
        # Sort keys as integers if they are '0', '1', etc.
        keys = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for key in keys:
            arr = f[key][:]
            if len(arr) >= seq_len:
                samples.append(arr[:seq_len])
            if len(samples) >= n:
                break
    if not samples:
        raise RuntimeError(f"No valid samples found in {h5_path!r}")
    return torch.tensor(np.stack(samples), dtype=torch.float32)


def save_codebook_chart(usage_m1: np.ndarray, usage_m4: np.ndarray, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        for ax, usage, title in zip(axes, [usage_m1, usage_m4], ["1-Bar (m1)", "4-Bar (m4)"]):
            ax.bar(range(len(usage)), usage, color="#4a9eff", edgecolor="none", alpha=0.85)
            ax.set_xlabel("Codebook Index")
            ax.set_ylabel("Usage Count")
            active = int((usage > 0).sum())
            usage_pct = active / len(usage) * 100
            ax.set_title(f"Codebook Utilization — {title}\n"
                         f"Active codes: {active}/{len(usage)} ({usage_pct:.1f}%)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  📊 Chart saved → {out_path}")
    except ImportError:
        print("  ⚠️  matplotlib not installed, skipping chart.")


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def inspect(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(MIDI_OUT, exist_ok=True)

    print(f"\n🔍 VQ-VAE Inspector")
    print(f"   Checkpoint : {args.checkpoint}")
    print(f"   H5 cache   : {args.data_cache}")
    print(f"   Num samples: {args.num_samples}")
    print(f"   Device     : {device}")
    print("=" * 60)

    # ── Load checkpoint ───────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"]
    
    # Infer dims from saved weights
    codebook_size, hidden_dim = state["vq_m1.embedding.weight"].shape
    # ResidualGRU stores the actual GRU in .gru
    input_dim = state["enc_m1.gru.weight_ih_l0"].shape[1]

    print(f"   input_dim={input_dim}, hidden_dim={hidden_dim}, codebook_size={codebook_size}")
    model = HierarchicalVQVAE(input_dim=input_dim, hidden_dim=hidden_dim, codebook_size=codebook_size)
    
    # Handle potentially missing decoder weights in older checkpoints (though we just added it)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        print(f"  ⚠️  Checkpoint might be missing Decoder weights: {e}")
        model.load_state_dict(state, strict=False)
        
    model.to(device).eval()

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n📂 Loading samples from H5 cache...")
    x = load_samples(args.data_cache, args.seq_len, args.num_samples).to(device)
    print(f"   Loaded {x.shape[0]} samples, shape: {x.shape}")

    # ── Run forward pass ──────────────────────────────────────────
    print("\n⚙️  Running VQ-VAE forward pass...")
    all_idx_m1 = []
    all_idx_m4 = []
    all_recon = []

    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(x), batch_size), desc="Inferencing", ncols=80):
            batch = x[i : i + batch_size]
            # Hierarchical returns (recon_x_denorm, q_m1, q_m4, total_loss, idx_m1, idx_m4)
            recon_x, q_m1, q_m4, _, idx_m1, idx_m4 = model(batch)

            all_idx_m1.append(idx_m1.cpu().numpy().reshape(len(batch), -1))
            all_idx_m4.append(idx_m4.cpu().numpy().reshape(len(batch), -1))
            all_recon.append(recon_x.cpu().numpy())

    idx_m1 = np.concatenate(all_idx_m1, axis=0)
    idx_m4 = np.concatenate(all_idx_m4, axis=0)
    recon = np.concatenate(all_recon, axis=0)

    # ── MIDI Reconstruction ───────────────────────────────────────
    print(f"\n🎵 Generating MIDI reconstructions for first {args.num_midi} samples...")
    for i in range(min(args.num_midi, len(x))):
        orig_data = x[i].cpu().numpy()
        recon_data = recon[i]
        
        # Round recon_data to integers for MIDI values
        recon_data = np.round(recon_data).astype(np.int16)
        
        orig_path = os.path.join(MIDI_OUT, f"sample_{i:02d}_orig.mid")
        recon_path = os.path.join(MIDI_OUT, f"sample_{i:02d}_recon.mid")
        
        octuple8d_to_midi(orig_data, orig_path)
        octuple8d_to_midi(recon_data, recon_path)

    # ── Statistics ────────────────────────────────────────────────
    usage_m1 = np.bincount(idx_m1.flatten(), minlength=codebook_size)
    usage_m4 = np.bincount(idx_m4.flatten(), minlength=codebook_size)

    def perplexity(usage):
        p = usage / usage.sum()
        p = p[p > 0]
        return float(np.exp(-(p * np.log(p)).sum()))

    pplx_m1 = perplexity(usage_m1)
    pplx_m4 = perplexity(usage_m4)

    # ── Save report ───────────────────────────────────────────────
    report = f"""OmniLLM-Muse VQ-VAE Inspection Report
======================================
Checkpoint : {args.checkpoint}
Epoch      : {ckpt.get('epoch', 'N/A')}
Train loss : {ckpt.get('loss', 'N/A')}

Model
  input_dim    = {input_dim}
  hidden_dim   = {hidden_dim}
  codebook_size = {codebook_size}

Reconstruction
  MSE Loss (Samples) : {np.mean((recon - x.cpu().numpy())**2):.6f}

Codebook Analysis
  1-Bar (m1) PPLX : {pplx_m1:.2f} ({int((usage_m1 > 0).sum())}/{codebook_size} active, {int((usage_m1 > 0).sum())/codebook_size*100:.1f}%)
  4-Bar (m4) PPLX : {pplx_m4:.2f} ({int((usage_m4 > 0).sum())}/{codebook_size} active, {int((usage_m4 > 0).sum())/codebook_size*100:.1f}%)

MIDI Outputs saved to: {MIDI_OUT}/
"""
    with open(f"{OUT_DIR}/report.txt", "w") as f:
        f.write(report)
    
    save_codebook_chart(usage_m1, usage_m4, f"{OUT_DIR}/codebook_usage.png")
    
    # Save the raw numpy arrays for deeper analysis if needed
    np.save(f"{OUT_DIR}/recon_x.npy", recon)
    np.save(f"{OUT_DIR}/indices_m1.npy", idx_m1)
    
    print(report)
    print(f"✅ All outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best/hierarchical_epoch10.pt")
    parser.add_argument("--data-cache", default="data/lmd_8d_best_cache.h5")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-midi", type=int, default=10, help="Number of MIDI pairs to generate")
    parser.add_argument("--seq-len", type=int, default=128)
    inspect(parser.parse_args())
