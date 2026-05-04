"""
H-VQ-VAE / Vanilla VQ-VAE Training Script
------------------------------------------
Consumes OmniConfig (SOLID Dependency Inversion).
Run:
  # smoke test (3 epochs, 100 files)
  python src/trainer/vqvae_trainer.py --mode smoke

  # full run
  python src/trainer/vqvae_trainer.py --mode full

  # custom JSON config
  python src/trainer/vqvae_trainer.py --config my_config.json
"""
import sys
import os
import random
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from miditok import Octuple
from symusic import Score
from concurrent.futures import ProcessPoolExecutor

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.vqvae import HierarchicalVQVAE
from models.vqvae_standard import VanillaVQVAE
from models.octuple import get_extractor
from config import OmniConfig, smoke_test_config, full_train_config, resolve_input_dim, best_8d_config, smoke_test_config_8d
from sklearn.cluster import MiniBatchKMeans


# ============================
# Parallel Helper
# ============================
def _parallel_tokenize(args):
    """Worker function for ProcessPoolExecutor."""
    path, token_mode = args
    # Import inside worker to avoid serializing complex objects
    from models.octuple import get_extractor
    import os
    # Re-verify src in path for worker processes
    import sys
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        
    extractor = get_extractor(token_mode)
    return extractor.extract(path)


# ============================
# Dataset
# ============================
class LMDOctupleDataset(Dataset):
    """
    Builds (and caches to H5) tokenized OctupleMIDI sequences.
    Accepts DataConfig — no direct I/O constants in this class.
    """
    def __init__(self, cfg, data_dir=None):
        self.seq_len = cfg.seq_len
        self.samples = []

        if data_dir and os.path.isdir(data_dir):
            self._load_from_dir(data_dir)
        else:
            if not os.path.exists(cfg.h5_cache):
                print(f"⚙️  H5 cache not found. Building from {cfg.lmd_dir} ...")
                self._build_cache(cfg)

            print(f"📂 Loading H5 dataset: {cfg.h5_cache}")
            with h5py.File(cfg.h5_cache, "r") as f:
                keys = list(f.keys())
                print(f"✅ {len(keys)} cached tracks found.")
                for k in tqdm(keys, desc="Loading H5 cache", ncols=100):
                    arr = f[k][:]
                    for start in range(0, len(arr) - self.seq_len, self.seq_len // 2):
                        self.samples.append(arr[start : start + self.seq_len])

        print(f"📊 Total training samples: {len(self.samples)}")

    def _load_from_dir(self, data_dir):
        """Loads pre-tokenized .npy files from a directory."""
        print(f"📂 Loading pre-tokenized samples from {data_dir}...")
        npy_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        if not npy_files:
            print(f"⚠️  No .npy files found in {data_dir}!")
            return

        for f in tqdm(npy_files, desc="Loading .npy files", ncols=100):
            try:
                arr = np.load(f)
                if len(arr) < self.seq_len:
                    continue
                for start in range(0, len(arr) - self.seq_len, self.seq_len // 2):
                    self.samples.append(arr[start : start + self.seq_len])
            except Exception:
                continue

    def _build_cache(self, cfg):
        all_midis = []
        for root, _, files in os.walk(cfg.lmd_dir):
            for f in files:
                if f.endswith((".mid", ".midi")):
                    all_midis.append(os.path.join(root, f))

        if cfg.max_files is not None:
            random.shuffle(all_midis)
            all_midis = all_midis[: cfg.max_files]

        num_files = len(all_midis)
        print(f"🎵 Parallel Tokenizing {num_files} MIDI files ({cfg.token_mode})...")
        
        # Use ProcessPoolExecutor for multi-core speedup
        # max_workers=None defaults to CPU count
        with h5py.File(cfg.h5_cache, "w") as fout:
            ok = skip = 0
            args_list = [(p, cfg.token_mode) for p in all_midis]
            
            with ProcessPoolExecutor() as executor:
                # Use executor.map to maintain order and simplify iteration
                # We wrap in tqdm for progress tracking
                for i, ids in enumerate(tqdm(executor.map(_parallel_tokenize, args_list), 
                                           total=num_files, desc="🔢 Parallel Extraction", ncols=100)):
                    if ids is not None and len(ids) >= self.seq_len:
                        fout.create_dataset(str(i), data=ids, compression="gzip")
                        ok += 1
                    else:
                        skip += 1
        print(f"✅ Cache built: {ok} tracks written, {skip} skipped.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx].copy()  # (L, 8)
        
        # ── Relative Bar Fix ───────────────────────────────────────
        # Subtract the starting bar index to make the sample position-invariant.
        # This prevents the VQ-VAE from trying to learn absolute bar indices.
        if sample.shape[1] >= 1:
            start_bar = sample[0, 0]
            sample[:, 0] -= start_bar
            
        return torch.tensor(sample, dtype=torch.float32)


# ============================
# Perplexity
# ============================
def compute_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return (-(probs * probs.log()).sum()).exp().item()


# ============================
# Model Factory (Factory Pattern)
# ============================
def build_model(cfg) -> nn.Module:
    if cfg.variant == "hierarchical":
        return HierarchicalVQVAE(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            codebook_size=cfg.codebook_size,
            commitment_cost=cfg.commitment_cost,
        )
    elif cfg.variant == "vanilla":
        return VanillaVQVAE(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            codebook_size=cfg.codebook_size,
        )
    else:
        raise ValueError(f"Unknown model variant: {cfg.variant!r}")


# ============================
# K-Means Initialization
# ============================
def kmeans_init_codebook(model, loader, device, mcfg, num_samples=2000):
    """
    Extract features from input and perform K-means to initialize codebooks.
    This helps prevent initial collapse and boosts Epoch 0 performance.
    """
    print(f"🧬 Performing K-means Initialization using {num_samples} samples...")
    
    model.eval()
    all_m1_feats = []
    
    samples_collected = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Gathering features for K-means", leave=False):
            x = batch.to(device)
            # Normalize as in forward pass
            x_norm = x.float() / 128.0 - 1.0
            
            # Encoder features for m1
            m1_out = model.enc_m1(x_norm)
            all_m1_feats.append(m1_out.reshape(-1, m1_out.size(-1)).cpu())
            
            samples_collected += x.size(0)
            if samples_collected >= num_samples:
                break
    
    m1_data = torch.cat(all_m1_feats, dim=0).numpy()
    
    # Init m1 vq
    print(f"   [m1] Clustering {m1_data.shape[0]} points into {mcfg.codebook_size} centers...")
    kmeans = MiniBatchKMeans(n_clusters=mcfg.codebook_size, batch_size=1024, n_init="auto")
    kmeans.fit(m1_data)
    model.vq_m1.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
    # Sync EMA parameters
    model.vq_m1.ema_w.data.copy_(model.vq_m1.embedding.weight.data)
    model.vq_m1.ema_cluster_size.data.fill_(1.0)

    # Init m4 (hierarchical only)
    if mcfg.variant == "hierarchical":
        all_m4_feats = []
        samples_collected = 0
        with torch.no_grad():
             for batch in loader:
                x = batch.to(device)
                x_norm = x.float() / 128.0 - 1.0
                m1_out = model.enc_m1(x_norm)
                m1_res = m1_out.transpose(1, 2)
                m4_out = F.elu(model.enc_m4_conv(m1_res)).transpose(1, 2)
                m4_out = model.enc_m4_ln(m4_out)
                all_m4_feats.append(m4_out.reshape(-1, m4_out.size(-1)).cpu())
                samples_collected += x.size(0)
                if samples_collected >= num_samples:
                    break
        m4_data = torch.cat(all_m4_feats, dim=0).numpy()
        print(f"   [m4] Clustering {m4_data.shape[0]} points into {mcfg.codebook_size} centers...")
        kmeans = MiniBatchKMeans(n_clusters=mcfg.codebook_size, batch_size=1024, n_init="auto")
        kmeans.fit(m4_data)
        model.vq_m4.embedding.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
        model.vq_m4.ema_w.data.copy_(model.vq_m4.embedding.weight.data)
        model.vq_m4.ema_cluster_size.data.fill_(1.0)
    
    print("✅ K-means Initialization complete.")


# ============================
# Training Loop
# ============================
def load_checkpoint_flexible(model, path, device):
    """Loads state_dict with key mapping for backward compatibility."""
    print(f"📂 Attempting flexible load from: {path}")
    state = torch.load(path, map_location=device)
    if "model_state" not in state:
        print("   ❌ Checkpoint lacks 'model_state'. Aborting load.")
        return 1
        
    old_sd = state["model_state"]
    new_sd = model.state_dict()
    
    # Mapping based on ResidualGRU and Conv1d refactor
    mapping = {
        "enc_m1.weight_ih_l0": "enc_m1.gru.weight_ih_l0",
        "enc_m1.weight_hh_l0": "enc_m1.gru.weight_hh_l0",
        "enc_m1.bias_ih_l0": "enc_m1.gru.bias_ih_l0",
        "enc_m1.bias_hh_l0": "enc_m1.gru.bias_hh_l0",
        "vq_m1.embedding.weight": "vq_m1.embedding.weight",
        "enc_m4.weight": "enc_m4_conv.weight",
        "enc_m4.bias": "enc_m4_conv.bias",
        "vq_m4.embedding.weight": "vq_m4.embedding.weight",
        "dec.weight_ih_l0": "dec.gru.weight_ih_l0",
        "dec.weight_hh_l0": "dec.gru.weight_hh_l0",
        "dec.bias_ih_l0": "dec.gru.bias_ih_l0",
        "dec.bias_hh_l0": "dec.gru.bias_hh_l0",
        "out_head.weight": "out_head.weight",
        "out_head.bias": "out_head.bias",
    }
    
    mapped_sd = {}
    for k, v in old_sd.items():
        if k in mapping:
            target_k = mapping[k]
            if target_k in new_sd and v.shape == new_sd[target_k].shape:
                mapped_sd[target_k] = v
            else:
                s1 = v.shape
                s2 = new_sd[target_k].shape if target_k in new_sd else "N/A"
                print(f"   ⚠️  Shape/Key mismatch for {k}: {s1} vs {s2}. skipping.")
        
    # Load what we can
    model.load_state_dict(mapped_sd, strict=False)
    
    missing = set(new_sd.keys()) - set(mapped_sd.keys())
    if missing:
        print(f"   ℹ️  {len(missing)} new parameters initialized from scratch (EMA buffers, LayerNorms, etc.)")
    
    print("✅ Checkpoint weights partially migrated.")
    return state.get("epoch", 0)

def train(omni_cfg: OmniConfig, resume_path: str = None, model_name: str = None, run_kmeans: bool = False, data_dir: str = None):
    dcfg = omni_cfg.data
    mcfg = omni_cfg.model
    tcfg = omni_cfg.training

    # Auto-resolve input_dim from token_mode if set to sentinel -1
    if mcfg.input_dim == -1:
        mcfg.input_dim = resolve_input_dim(dcfg.token_mode)

    device = tcfg.resolved_device()

    print(f"\n🚀 OmniLLM-Muse VQ-VAE Trainer")
    print(f"   Variant   : {mcfg.variant}")
    print(f"   Token mode: {dcfg.token_mode} (input_dim={mcfg.input_dim})")
    print(f"   Device    : {device}")
    print(f"   Epochs    : {tcfg.num_epochs}")
    print(f"   Max files : {dcfg.max_files}")
    print("=" * 60)

    full_dataset = LMDOctupleDataset(dcfg, data_dir=data_dir)
    if len(full_dataset) == 0:
        raise RuntimeError("❌ Dataset is empty — check lmd_dir and max_files.")

    val_size = int(0.05 * len(full_dataset))
    val_size = max(1, val_size) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    loader = DataLoader(
        train_ds,
        batch_size=tcfg.batch_size,
        shuffle=True,
        num_workers=min(tcfg.num_workers, 4),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=tcfg.batch_size,
        shuffle=False,
        num_workers=min(tcfg.num_workers, 4),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = build_model(mcfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg.num_epochs)

    # ── Resume Training (Flexible) ───────────────────────────────
    start_epoch = 1
    if resume_path:
        if not os.path.exists(resume_path):
            print(f"⚠️  Resume checkpoint not found: {resume_path}")
        else:
            last_epoch = load_checkpoint_flexible(model, resume_path, device)
            start_epoch = last_epoch + 1
            print(f"   Starting from epoch {start_epoch}")
    
    # ── K-Means Initialization ─────────────────────────────────────
    if run_kmeans and start_epoch == 1:
        kmeans_init_codebook(model, loader, device, mcfg)

    print(f"📐 Params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"📦 Batches: {len(loader)}")
    print("=" * 60)

    os.makedirs(tcfg.checkpoint_dir, exist_ok=True)

    try:
        for epoch in range(start_epoch, start_epoch + tcfg.num_epochs):
            model.train()
            epoch_loss = 0.0
            perp_total = 0.0

            pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{tcfg.num_epochs}", unit="batch", ncols=110)
            for batch in pbar:
                x = batch.to(device)
                optimizer.zero_grad()

                out = model(x)
                
                if mcfg.variant == "hierarchical":
                    # (recon_x_denorm, q_m1, q_m4, total_loss, idx_m1, idx_m4)
                    recon_x, q_m1, q_m4, loss, idx1, idx4 = out
                    p1 = compute_perplexity(idx1, mcfg.codebook_size)
                    p4 = compute_perplexity(idx4, mcfg.codebook_size)
                    p = (p1 + p4) / 2
                else:
                    # (recon_x, loss, indices)
                    recon_x, loss, idx = out
                    p = compute_perplexity(idx, mcfg.codebook_size)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()
                perp_total += p

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "pplx": f"{p:.1f}"})

            scheduler.step()
            n = len(loader) if len(loader) > 0 else 1
            avg_train_loss = epoch_loss / n
            avg_train_perp = perp_total / n
            
            # Validation Loop
            model.eval()
            val_loss_total = 0.0
            val_perp_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch.to(device)
                    out = model(x)
                    if mcfg.variant == "hierarchical":
                        recon_x, q_m1, q_m4, loss, idx1, idx4 = out
                        p1 = compute_perplexity(idx1, mcfg.codebook_size)
                        p4 = compute_perplexity(idx4, mcfg.codebook_size)
                        p = (p1 + p4) / 2
                    else:
                        recon_x, loss, idx = out
                        p = compute_perplexity(idx, mcfg.codebook_size)
                    val_loss_total += loss.item()
                    val_perp_total += p
                    
            n_val = len(val_loader) if len(val_loader) > 0 else 1
            avg_val_loss = val_loss_total / n_val
            avg_val_perp = val_perp_total / n_val
            
            lr_now = scheduler.get_last_lr()[0]
            print(f"  📈 epoch={epoch:02d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | train_pplx={avg_train_perp:.2f} | val_pplx={avg_val_perp:.2f} | lr={lr_now:.2e}")

            if epoch % tcfg.save_every == 0 or epoch == (start_epoch + tcfg.num_epochs - 1):
                name = model_name if model_name else mcfg.variant
                ckpt = os.path.join(tcfg.checkpoint_dir, f"{name}_epoch{epoch:02d}.pt")
                torch.save({"epoch": epoch, "model_state": model.state_dict(), "loss": avg_train_loss}, ckpt)
                print(f"  💾 Checkpoint saved → {ckpt}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user. Saving current weights...")
        name = model_name if model_name else mcfg.variant
        ckpt = os.path.join(tcfg.checkpoint_dir, f"{name}_interrupted.pt")
        torch.save({"epoch": epoch, "model_state": model.state_dict()}, ckpt)
        print(f"  💾 Emergency checkpoint saved → {ckpt}")

    print("\n✅ Training session complete!")


# ============================
# Entry Point
# ============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OmniLLM-Muse VQ-VAE Trainer")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mode", choices=["smoke", "full", "best8d", "smoke8d"], default="smoke",
                       help="Use a pre-baked config profile (default: smoke)")
    group.add_argument("--config", type=str, help="Path to a JSON config file")
    parser.add_argument("--variant", choices=["hierarchical", "vanilla"],
                        default=None, help="Override model variant")
    parser.add_argument("--hidden_dim", type=int, help="Override hidden dimension")
    parser.add_argument("--codebook_size", type=int, help="Override codebook size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--max_files", type=int, help="Override maximum number of MIDI files")
    parser.add_argument("--model_name", type=str, help="Custom name for the model (affects checkpoint filename)")
    parser.add_argument("--resume", type=str, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--force", action="store_true", help="Force rebuild of H5 cache")
    parser.add_argument("--kmeans", action="store_true", help="Run K-means initialization (Epoch 1 only)")
    parser.add_argument("--data_dir", type=str, help="Load pre-tokenized .npy files from this directory")
    args = parser.parse_args()

    if args.config:
        cfg = OmniConfig.load(args.config)
    elif args.mode == "full":
        cfg = full_train_config()
    elif args.mode == "best8d":
        cfg = best_8d_config()
    elif args.mode == "smoke8d":
        cfg = smoke_test_config_8d()
    else:
        cfg = smoke_test_config()

    if args.variant:
        cfg.model.variant = args.variant
    if args.hidden_dim:
        cfg.model.hidden_dim = args.hidden_dim
    if args.codebook_size:
        cfg.model.codebook_size = args.codebook_size
    if args.epochs:
        cfg.training.num_epochs = args.epochs
    if args.max_files:
        cfg.data.max_files = args.max_files
    
    # ── Force rebuild cache ──────────────────────────────────────
    if args.force and os.path.exists(cfg.data.h5_cache):
        print(f"🔥 Force rebuilding cache: deleting {cfg.data.h5_cache}")
        os.remove(cfg.data.h5_cache)

    print(cfg)
    train(cfg, resume_path=args.resume, model_name=args.model_name, run_kmeans=args.kmeans, data_dir=args.data_dir)
