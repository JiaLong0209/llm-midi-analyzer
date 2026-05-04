import json
import os
from tqdm import tqdm

def main():
    # resolve paths relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md5_path = os.path.join(project_root, 'data/md5_to_paths.json')
    midicaps_path = os.path.join(project_root, 'data/MidiCaps/train.json')
    output_path = os.path.join(project_root, 'data/mapped_midicaps.jsonl')
    
    print(f"Loading MD5 mappings from {md5_path}...")
    with open(md5_path, 'r') as f:
        md5_to_paths = json.load(f)
        
    print(f"Loaded {len(md5_to_paths)} MD5-to-path mappings.")
    
    midicaps_data = []
    print(f"Loading MidiCaps dataset from {midicaps_path}...")
    try:
        with open(midicaps_path, 'r') as f:
            for line in f:
                if line.strip():
                    midicaps_data.append(json.loads(line))
        print(f"Loaded {len(midicaps_data)} MidiCaps entries.")
    except Exception as e:
        print(f"Error reading train.json: {e}")
        return
    
    mapped_count = 0
    with open(output_path, 'w') as out_f:
        for entry in tqdm(midicaps_data, desc="Mapping entries"):
            md5_raw = entry.get('location', '')
            md5_clean = md5_raw.split('/')[-1].replace('.mid', '')
            if md5_clean in md5_to_paths:
                midi_file_path = md5_to_paths[md5_clean]
                out_entry = {
                    "caption": entry.get("caption", ""),
                    "midi_path": midi_file_path,
                    "location": md5_clean,
                    "tempo": entry.get("tempo", ""),
                    "key": entry.get("key", ""),
                    "chord_summary": entry.get("chord_summary", "")
                }
                out_f.write(json.dumps(out_entry) + '\n')
                mapped_count += 1
                
    print(f"Successfully mapped {mapped_count} entries.")
    print(f"Saved mapped dataset to {output_path}")

if __name__ == "__main__":
    main()
