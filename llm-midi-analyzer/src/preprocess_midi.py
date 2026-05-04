import os
import sys
import argparse
import numpy as np
import concurrent.futures
import random
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Helper for parallel map - needs to be global for pickling
def _parallel_tokenize(args):
    """Worker function for ProcessPoolExecutor."""
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    path, token_mode, out_path = args
    if os.path.exists(out_path):
        return True # Skip
    
    try:
        from models.octuple import get_extractor
        extractor = get_extractor(token_mode)
        ids = extractor.extract(path)
        if ids is not None:
            np.save(out_path, ids)
            return True
    except Exception as e:
        print(f"Error processing {path}: {e}")
    return False

def main():
    parser = argparse.ArgumentParser(description="Parallel MIDI to Octuple Preprocessor")
    parser.add_argument("--input", required=True, help="Directory containing MIDI files")
    parser.add_argument("--output", required=True, help="Output directory for .npy files")
    parser.add_argument("--mode", default="octuple_8d", help="Tokenization mode (default: octuple_8d)")
    parser.add_argument("--max_files", type=int, help="Limit number of files to process")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of parallel workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    print(f"🔍 Searching for MIDI files in {args.input}...")
    all_midis = []
    for root, _, files in os.walk(args.input):
        for f in files:
            if f.endswith((".mid", ".midi")):
                all_midis.append(os.path.join(root, f))
    
    if args.max_files:
        random.shuffle(all_midis)
        all_midis = all_midis[:args.max_files]
        
    num_total = len(all_midis)
    print(f"🎵 Found {num_total} MIDI files.")
    
    # Prepare task arguments
    tasks = []
    for i, p in enumerate(all_midis):
        # Use original MD5 basename to maintain tracking with descriptions
        out_name = os.path.basename(p).replace('.mid', '.npy').replace('.midi', '.npy')
        tasks.append((p, args.mode, os.path.join(args.output, out_name)))
        
    print(f"⚙️  Starting parallel extraction using {args.workers} workers...")
    ok = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Wrap the generator in list() if needed, but executor.map is lazy
        for res in tqdm(executor.map(_parallel_tokenize, tasks), total=num_total, desc="Converting"):
            if res:
                ok += 1
                
    print(f"✅ Finished! Extracted {ok}/{num_total} files to {args.output}")

if __name__ == "__main__":
    main()
