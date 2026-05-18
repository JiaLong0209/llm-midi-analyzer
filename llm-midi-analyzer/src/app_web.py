import os
import sys
import time

# --- Monkey Patch for torchao/transformers compatibility ---
import torch
if not hasattr(torch, 'int1'):
    torch.int1 = torch.int8
# ---------------------------------------------------------

import shutil
import json
import glob
from datetime import datetime
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional
import networkx as nx

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from generate_analysis import load_inference_model, run_inference
from config import AppWorkflowConfig
from music21_analyzer import Music21MidiAnalyzer
from gemini_service import GeminiService
from graph_rag import MusicKnowledgeGraph
from cag import CAGKV

app = FastAPI(title="LLM-MIDI-Analyzer Web")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import uuid

app_cfg = AppWorkflowConfig()
model_cache = {}
session_store = {}

@app.on_event("startup")
def startup_event():
    print("🚀 Booting LLM-MIDI-Analyzer Web Server...")
    print("Loading Heavy Models (Llama)...")
    try:
        # Override to use CPU or CUDA based on environment, assume CUDA here for performance
        tokenizer, llm, adapter, config = load_inference_model(
            checkpoint_path="checkpoints/adapter/musicbert_epoch10.pt",
            mode_override="musicbert",
            device="cuda",
            dtype_override="float16"
        )
        model_cache["tokenizer"] = tokenizer
        model_cache["llm"] = llm
        model_cache["adapter"] = adapter
        model_cache["config"] = config
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading heavy models on CUDA: {e}. Trying CPU fallback...")
        try:
            tokenizer, llm, adapter, config = load_inference_model(
                checkpoint_path="checkpoints/adapter/musicbert_epoch10.pt",
                mode_override="musicbert",
                device="cpu",
                dtype_override="float32"
            )
            model_cache["tokenizer"] = tokenizer
            model_cache["llm"] = llm
            model_cache["adapter"] = adapter
            model_cache["config"] = config
            print("✅ Models loaded successfully on CPU!")
        except Exception as cpu_err:
            print(f"⚠️ CPU fallback also failed: {cpu_err}. Analysis will fail if triggered.")

@app.post("/api/analyze")
async def analyze_midi(
    file: UploadFile = File(...),
    model_name: str = Form("gemini-3.1-flash-lite"),
    temperature: float = Form(0.3),
    enable_music21: bool = Form(True),
    include_detailed_tracks: bool = Form(False),
    enable_rag: bool = Form(True),
    enable_cag: bool = Form(True),
    start_measure: Optional[int] = Form(None),
    end_measure: Optional[int] = Form(None),
    user_prompt: Optional[str] = Form(None),
    rag_lang: Optional[str] = Form("zh-tw"),
):
    if not file.filename.endswith(('.mid', '.midi')):
        return JSONResponse(status_code=400, content={"error": "File must be a MIDI file."})

    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), "../output/web_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"🎵 Received {file.filename} for analysis.")
    
    # 1. Llama Inference
    llama_out = ""
    if "llm" in model_cache:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() and next(model_cache["llm"].parameters()).is_cuda else "cpu"
            print(f"🤖 Running local Llama inference on device: {device}...")
            
            # Move adapter to the correct device to avoid RuntimeError device mismatch
            model_cache["adapter"] = model_cache["adapter"].to(device)
            
            llama_out = run_inference(
                midi_path=file_path,
                tokenizer=model_cache["tokenizer"],
                llm=model_cache["llm"],
                adapter=model_cache["adapter"],
                acfg=model_cache["config"],
                device=device
            )
        except Exception as e:
            print(f"⚠️ Local Llama inference failed: {e}. Skipping local inference.")
            llama_out = f"Skipped Model Inference due to error: {e}"
    else:
        llama_out = "Skipped Model Inference due to missing model."

    # 2. Music21
    m21_data = {}
    
    start_measure_int = None
    if start_measure is not None:
        try:
            start_measure_int = int(start_measure)
            if start_measure_int <= 0:
                start_measure_int = None
        except:
            pass
            
    end_measure_int = None
    if end_measure is not None:
        try:
            end_measure_int = int(end_measure)
            if end_measure_int <= 0:
                end_measure_int = None
        except:
            pass

    if enable_music21:
        m21_analyzer = Music21MidiAnalyzer()
        m21_data = m21_analyzer.analyze_file(file_path, start_measure=start_measure_int, end_measure=end_measure_int)
        # Remove detailed_tracks from context if disabled (too long)
        if not include_detailed_tracks:
            m21_data.pop('detailed_tracks', None)

    # 3. Graph RAG
    gemini = GeminiService(model_name=model_name)
    
    llama_snippet = llama_out[:800]
    m21_snippet = str(m21_data)[:800]
    extracted_keywords = gemini.extract_music_keywords(llama_snippet, m21_snippet)
    
    if not extracted_keywords:
        extracted_keywords = ["Music", "MIDI"]

    kg = MusicKnowledgeGraph(use_web_search=enable_rag, lang=rag_lang)
    rag_context = kg.get_analysis_context(extracted_keywords) if enable_rag else ""
    
    # Extract graph nodes and edges for vis.js
    nodes = []
    edges = []
    for node, data in kg.graph.nodes(data=True):
        nodes.append({
            "id": node,
            "label": node,
            "group": data.get("type", "other"),
            "title": data.get("extract", "No description available.") # Tooltip
        })
    
    for u, v, data in kg.graph.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "label": data.get("relation", "")
        })

    # 4. CAG
    cag_context = ""
    if enable_cag:
        cag = CAGKV(source_dir=os.path.join(os.path.dirname(__file__), "../books"), cache_dir="kv_cache")
        cag_context = cag._build_doc_text(extracted_keywords)

    # 5. Final Report
    final_report = gemini.generate_final_report(
        llama_analysis=llama_out,
        music21_data=m21_data,
        rag_context=rag_context,
        cag_context=cag_context,
        start_measure=start_measure_int,
        end_measure=end_measure_int,
        user_prompt=user_prompt
    )
    if not final_report or final_report.startswith("[!"):
        # Fallback if the selected model fails
        print(f"⚠️ Model {model_name} failed. Retrying with gemini-3.1-flash-lite...")
        gemini_fallback = GeminiService(model_name="gemini-3.1-flash-lite")
        final_report = gemini_fallback.generate_final_report(
            llama_analysis=llama_out,
            music21_data=m21_data,
            rag_context=rag_context,
            cag_context=cag_context,
            start_measure=start_measure_int,
            end_measure=end_measure_int,
            user_prompt=user_prompt
        )

    session_id = str(uuid.uuid4())
    full_context = f"【Final Report】\n{final_report}\n\n【CAG Textbook Context】\n{cag_context}\n\n【GraphRAG Context】\n{rag_context}"
    session_store[session_id] = {
        "context": full_context,
        "history": [],
        "kg": kg # Store graph instance for dynamic updates
    }

    # Save analysis to output/analysis/
    midi_stem = os.path.splitext(file.filename)[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join(os.path.dirname(__file__), "../output/analysis", f"{midi_stem}_{ts}")
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, "final_report.md"), "w", encoding="utf-8") as f:
        f.write(final_report)
    with open(os.path.join(analysis_dir, "music21.json"), "w", encoding="utf-8") as f:
        json.dump(m21_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(analysis_dir, "llama_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(llama_out)
    with open(os.path.join(analysis_dir, "rag_context.txt"), "w", encoding="utf-8") as f:
        f.write(rag_context)
    with open(os.path.join(analysis_dir, "cag_context.txt"), "w", encoding="utf-8") as f:
        f.write(cag_context)
    graph_data = {
        "nodes": nodes,
        "edges": edges
    }
    with open(os.path.join(analysis_dir, "graph_data.json"), "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(analysis_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"midi_file": file.filename, "timestamp": ts, "model": model_name, "session_id": session_id}, f, ensure_ascii=False, indent=2)
    print(f"💾 Analysis saved to: {analysis_dir}")

    return {
        "status": "success",
        "session_id": session_id,
        "llama_analysis": llama_out,
        "music21_data": m21_data,
        "final_report": final_report,
        "graph_data": graph_data
    }

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    model_name: Optional[str] = "gemini-3.1-flash-lite"
    enable_chat_rag: Optional[bool] = False

@app.post("/api/chat")
async def chat_with_ai(request: ChatRequest):
    gemini = GeminiService(model_name=request.model_name or "gemini-3.1-flash-lite")
    new_graph_data = None
    
    if request.session_id and request.session_id in session_store:
        session_data = session_store[request.session_id]
        context = session_data["context"]
        
        # Dynamic RAG on chat message if enabled
        if request.enable_chat_rag:
            print(f"   [Chat RAG] Searching web for: {request.message}")
            try:
                grounding_info = gemini.search_with_grounding(request.message)
                if grounding_info and grounding_info.get('extract'):
                    context += f"\n\n【Dynamic Web Search Context for '{request.message}'】\n{grounding_info['extract']}"
                    # Keep session context updated with new info
                    session_data["context"] = context
                    
                    # --- NEW: Dynamic Graph Update ---
                    if "kg" in session_data:
                        # Extract keywords from the search result and user message
                        new_concepts = gemini.extract_music_keywords(request.message, grounding_info.get('extract', ''))
                        if new_concepts:
                            new_graph_data = session_data["kg"].add_new_concepts(new_concepts)
                            print(f"   [Chat RAG] Graph updated with {len(new_graph_data['nodes'])} new nodes.")
            except Exception as e:
                print(f"   [Chat RAG] Search failed: {e}")
                
        reply = gemini.chat_with_context(request.message, session_data["history"], context)
        session_data["history"].append({"user": request.message, "ai": reply})
    else:
        # Fallback to general chat if no session
        context = "General Musicology Context: The user is asking general questions without providing a specific MIDI file."
        if request.enable_chat_rag:
            try:
                grounding_info = gemini.search_with_grounding(request.message)
                if grounding_info and grounding_info.get('extract'):
                    context += f"\n\n【Web Search Context】\n{grounding_info['extract']}"
            except Exception:
                pass
        reply = gemini.chat_with_context(request.message, [], context)
        
    return {"reply": reply, "new_graph_data": new_graph_data}

@app.get("/api/midi-samples")
async def list_midi_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), "../midi")
    if not os.path.exists(samples_dir):
        return {"samples": []}
    files = [f for f in os.listdir(samples_dir) if f.endswith(('.mid', '.midi'))]
    return {"samples": files}

@app.get("/api/midi-samples/{filename}")
async def get_midi_sample(filename: str):
    samples_dir = os.path.join(os.path.dirname(__file__), "../midi")
    file_path = os.path.join(samples_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/midi")

@app.get("/api/analyses")
async def list_analyses():
    """List all saved analyses from output/analysis/ sorted chronologically by timestamp (newest first)"""
    base = os.path.join(os.path.dirname(__file__), "../output/analysis")
    analyses = []
    if os.path.exists(base):
        for meta_path in glob.glob(f"{base}/*/meta.json"):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                folder = os.path.basename(os.path.dirname(meta_path))
                analyses.append({
                    "id": folder,
                    "midi_file": meta.get("midi_file", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "model": meta.get("model", ""),
                })
            except Exception:
                pass
        
        # Sort chronologically descending
        analyses.sort(key=lambda x: x["timestamp"], reverse=True)
        
    return {"analyses": analyses}

@app.get("/api/analyses/{analysis_id}/load")
async def load_analysis(analysis_id: str):
    """Load a saved analysis and create a chat session."""
    base = os.path.join(os.path.dirname(__file__), "../output/analysis", analysis_id)
    try:
        with open(os.path.join(base, "final_report.md"), "r", encoding="utf-8") as f:
            final_report = f.read()
        with open(os.path.join(base, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            
        llama_out = ""
        llama_path = os.path.join(base, "llama_analysis.txt")
        if os.path.exists(llama_path):
            with open(llama_path, "r", encoding="utf-8") as f:
                llama_out = f.read()
                
        m21_data = {}
        m21_path = os.path.join(base, "music21.json")
        if os.path.exists(m21_path):
            with open(m21_path, "r", encoding="utf-8") as f:
                try:
                    m21_data = json.load(f)
                except Exception:
                    pass
                    
        rag_context = ""
        rag_path = os.path.join(base, "rag_context.txt")
        if os.path.exists(rag_path):
            with open(rag_path, "r", encoding="utf-8") as f:
                rag_context = f.read()
                
        cag_context = ""
        cag_path = os.path.join(base, "cag_context.txt")
        if os.path.exists(cag_path):
            with open(cag_path, "r", encoding="utf-8") as f:
                cag_context = f.read()
                
        graph_data = {"nodes": [], "edges": []}
        graph_path = os.path.join(base, "graph_data.json")
        if os.path.exists(graph_path):
            with open(graph_path, "r", encoding="utf-8") as f:
                try:
                    graph_data = json.load(f)
                except Exception:
                    pass
                
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "Analysis not found."})

    # Create new session from loaded context
    session_id = str(uuid.uuid4())
    session_store[session_id] = {
        "context": f"【Final Report】\n{final_report}\n\n【Llama Analysis】\n{llama_out}\n\n【CAG Textbook Context】\n{cag_context}\n\n【GraphRAG Context】\n{rag_context}",
        "history": []
    }
    return {
        "session_id": session_id,
        "final_report": final_report,
        "midi_file": meta.get("midi_file", ""),
        "timestamp": meta.get("timestamp", ""),
        "llama_analysis": llama_out,
        "music21": m21_data,
        "rag_context": rag_context,
        "cag_context": cag_context,
        "graph_data": graph_data,
    }

# Mount static files at root — MUST be last
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app_web:app", host="0.0.0.0", port=8081, reload=True)
