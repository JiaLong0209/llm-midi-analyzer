"""
app.py — OmniLLM-Muse Unified Entry Point
==========================================
Handles MIDI selection, multi-agent analysis (Llama + Music21),
RAG query rewriting, and Gemini final report generation.
"""

import os
import sys
import time

# --- Monkey Patch for torchao/transformers compatibility ---
import torch
if not hasattr(torch, 'int1'):
    torch.int1 = torch.int8
# ---------------------------------------------------------

import questionary
from datetime import datetime
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Optimize VRAM allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_analysis import load_inference_model, run_inference
from config import AppWorkflowConfig
from music21_analyzer import Music21MidiAnalyzer
from gemini_service import GeminiService



# ──────────────────────────────────────────────────────────────────────
# UI & Workflow
# ──────────────────────────────────────────────────────────────────────
def select_midi_file(midi_dir: str) -> str:
    if not os.path.exists(midi_dir):
        print(f"❌ Error: MIDI directory not found: {midi_dir}")
        sys.exit(1)
        
    files = [f for f in os.listdir(midi_dir) if f.endswith(".mid") or f.endswith(".midi")]
    if not files:
        print(f"❌ No MIDI files found in {midi_dir}")
        sys.exit(1)
        
    choice = questionary.select(
        "🎹 Select a MIDI file to analyze:",
        choices=sorted(files)
    ).ask()
    
    return os.path.join(midi_dir, choice)

import json

def save_analysis(midi_path: str, llama_out: str, m21_out: dict, keywords: list, rag_context: str, cag_context: str, final_report: str, out_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    midi_name = os.path.splitext(os.path.basename(midi_path))[0]
    run_dir = os.path.join(out_dir, f"{midi_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 1. Save music21.json
    if m21_out:
        json_path = os.path.join(run_dir, "music21.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(m21_out, f, indent=2, ensure_ascii=False)
            
    # 2. Save final_report.md
    md_path = os.path.join(run_dir, "final_report.md")
    content = f"# 🎼 Analysis Record: {midi_name}\n"
    content += f"**Date/Time:** {timestamp}\n"
    content += f"**File Path:** `{midi_path}`\n\n"
    
    content += "## 1. Llama 1B Deep Representation Output\n"
    content += f"> {llama_out.strip()}\n\n"
    
    content += "## 2. RAG & CAG Search\n"
    content += f"- **Extracted Keywords:** `{', '.join(keywords)}`\n"
    if rag_context:
        content += f"- **GraphRAG Context:**\n<details><summary>Click to expand</summary>\n\n```text\n{rag_context}\n```\n</details>\n\n"
    if cag_context:
        content += f"- **CAG (PDF) Context:**\n<details><summary>Click to expand</summary>\n\n```text\n{cag_context}\n```\n</details>\n\n"
    
    content += "## 3. Gemini Final Synthesized Report\n"
    content += f"{final_report}\n"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"\n💾 Saved full analysis to directory: {run_dir}/")

def main():
    print(f"\n{'='*60}")
    print("🎼 OMNILLM-MUSE: AI MUSIC ANALYST (Multi-Agent)")
    print(f"{'='*60}\n")

    # 1. Configuration
    app_cfg = AppWorkflowConfig()
    MIDI_DIR = "midi"
    CHECKPOINT = "checkpoints/adapter/musicbert_epoch10.pt"
    MODE = "musicbert" 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = "float16"

    # 2. Select Language
    lang = questionary.select(
        "🌐 Select analysis language:",
        choices=[
            questionary.Choice("English", "en"),
            questionary.Choice("繁體中文 (Traditional Chinese)", "zh")
        ]
    ).ask()

    # 3. Select MIDI
    midi_path = select_midi_file(MIDI_DIR)
    
    # 4. Phase 1: Llama 1B (Deep Feature Extraction)
    import time
    p1_start = time.time()
    print(f"\n🤖 [Phase 1] Initializing Llama 1B (Mode: {MODE.upper()})...")
    tokenizer, llm, adapter, config = load_inference_model(
        checkpoint_path=CHECKPOINT,
        mode_override=MODE,
        device=DEVICE,
        dtype_override=DTYPE,
        lang=lang
    )
    
    print(f"🚀 Running Neural Analysis: {os.path.basename(midi_path)}...")
    llama_analysis = run_inference(
        midi_path=midi_path,
        tokenizer=tokenizer,
        llm=llm,
        adapter=adapter,
        acfg=config,
        device=DEVICE,
        lang=lang
    )
    print(f"✅ Phase 1 completed in {time.time() - p1_start:.2f}s")

    # 5. Phase 2: Music21 (Symbolic Extraction)
    p2_start = time.time()
    music21_data = {}
    if app_cfg.enable_music21:
        print("\n🎻 [Phase 2] Running Music21 Symbolic Extraction...")
        m21_analyzer = Music21MidiAnalyzer()
        music21_data = m21_analyzer.analyze_file(midi_path)
        print(f"✅ Phase 2 completed in {time.time() - p2_start:.2f}s")
    else:
        print("\n⏭️  Skipping Music21 Extraction (disabled in config).")

    # 6. Phase 3: Gemini Synthesis & Post-RAG
    p3_start = time.time()
    rag_context = ""
    cag_context = ""
    extracted_keywords = []
    final_report = llama_analysis
    if app_cfg.enable_final_report:
        print("\n🧠 [Phase 3] Booting Gemini API for Synthesis & Post-RAG...")
        gemini = GeminiService(model_name=app_cfg.llm_model_name)
        
        # 7a. Extract Keywords based on Llama + Music21 findings
        print("   -> Extracting Theoretical Keywords for RAG/CAG...")
        llama_snippet = llama_analysis[:800]
        m21_snippet = json.dumps(music21_data, ensure_ascii=False)[:800] if music21_data else "{}"
        extracted_keywords = gemini.extract_music_keywords(llama_snippet, m21_snippet)
        print(f"   -> 🔎 Keywords: {extracted_keywords}")
        
        # 7b. Graph RAG Retrieval
        print("   -> 🌐 Searching GraphRAG (Wikipedia)...")
        from graph_rag import MusicKnowledgeGraph
        kg = MusicKnowledgeGraph(use_web_search=True)
        
        # Try loading from cache first to save time
        rag_context = ""
        if app_cfg.graph_cache_enabled and kg.load_from_cache():
            rag_context = kg._format_context(extracted_keywords, []) # Simplified reload
        else:
            rag_context = kg.get_analysis_context(extracted_keywords)
        
        # [NEW] Save Graph Visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        midi_name = os.path.splitext(os.path.basename(midi_path))[0]
        run_dir = os.path.join(app_cfg.output_dir, f"{midi_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        viz_path = os.path.join(run_dir, "knowledge_graph.png")
        kg.visualize(viz_path, show_desc=app_cfg.graph_viz_show_desc)
        
        # 7c. CAG (Local PDF) Retrieval
        print("   -> 📚 Searching CAG (Local Textbooks)...")
        from cag import CAGKV
        cag = CAGKV(source_dir=os.path.join(os.path.dirname(__file__), "../books"), cache_dir="kv_cache")
        # We bypass KV precomputation and just extract the text
        cag_context = cag._build_doc_text(extracted_keywords)
        
        # 7d. Generate Final Report
        print("   -> ✍️  Drafting final synthesized report...")
        final_report = gemini.generate_final_report(
            llama_analysis=llama_analysis,
            music21_data=music21_data,
            rag_context=rag_context,
            cag_context=cag_context
        )
        
        print("\n✨==========================================================")
        print("📜 FINAL MULTI-AGENT ANALYSIS REPORT")
        print("============================================================")
        print(final_report)
        print("============================================================\n")
        
        print(f"✅ Phase 3 completed in {time.time() - p3_start:.2f}s")
        
        # 8. Save
        save_analysis(
            midi_path=midi_path,
            llama_out=llama_analysis,
            m21_out=music21_data,
            keywords=extracted_keywords,
            rag_context=rag_context,
            cag_context=cag_context,
            final_report=final_report,
            out_dir=app_cfg.output_dir
        )
        
        # 9. Interactive Chat Loop
        print("\n💬 進入互動式諮詢模式 (Interactive Consultation Mode)")
        print("您可以針對這首曲子的功能和聲、作曲技巧，或報告中的內容進行提問。(輸入 'exit' 或 'q' 離開)")
        
        chat_history = []
        full_context = f"【Final Report】\n{final_report}\n\n【CAG Textbook Context】\n{cag_context}\n\n【GraphRAG Context】\n{rag_context}"
        
        while True:
            user_query = questionary.text("🧑‍🎵 Composer:").ask()
            if not user_query or user_query.strip().lower() in ['exit', 'q', 'quit']:
                break
                
            print("🤖 AI Analyst is thinking...")
            reply = gemini.chat_with_context(user_query, chat_history, full_context)
            print(f"\n🎼 AI:\n{reply}\n")
            
            chat_history.append({"user": user_query, "ai": reply})

if __name__ == "__main__":
    main()
