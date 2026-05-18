"""
generate_analysis_v3_graph_rag.py - 使用 Graph RAG + PDF CAG 的推理腳本
Graph RAG : 即時從 Wikipedia 網路搜尋，動態建立知識圖谱
CAG       : 從 CAG_source/ 資料夾的 PDF 文件中提取並注入上下文
"""

import sys
import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from config import AdapterConfig
from models.adapters import AdapterFactory
from models.octuple import get_extractor

# ── 新版模組 ──────────────────────────────────
from graph_rag import MusicKnowledgeGraph
from cag import CAGKV


# ──────────────────────────────────────────────
# MIDI 特徵偵測
# ──────────────────────────────────────────────

def detect_midi_features(tokens: np.ndarray) -> list:
    """
    從 OctupleMIDI tokens 偵測音樂特徵。

    tokens shape: (T, 8)
    維度: [Bar, Position, Program, Pitch, Duration, Velocity, TimeSig, Tempo]
    """
    features = []

    if len(tokens) == 0:
        return ["Unknown"]

    # 特徵 1：聲部織體
    programs = np.unique(tokens[:, 2])
    features.append("Polyphony" if len(programs) >= 3 else "Homophony")

    # 特徵 2：主要樂器
    main_program = int(np.median(tokens[:, 2]))
    if main_program <= 3:
        features.append("Keyboard")
    elif 40 <= main_program <= 43:
        features.append("Strings")
    else:
        features.append("WindInstruments")

    # 特徵 3：速度
    avg_tempo_bucket = float(np.mean(tokens[:, 7]))
    tempo_bpm = 60 + avg_tempo_bucket * 2.5
    if tempo_bpm > 140:
        features.append("FastTempo")
    elif tempo_bpm < 80:
        features.append("SlowTempo")
    else:
        features.append("ModerateTempo")
    # 特徵 4：形式重複性
    half = len(tokens) // 2
    first_mean  = float(np.mean(tokens[:half, 3]))
    second_mean = float(np.mean(tokens[half:, 3]))
    features.append("RepetitiveForm" if abs(first_mean - second_mean) < 5 else "DevelopingForm")

    # 特徵 5：調性
    pitches = tokens[:, 3].astype(int)
    pitch_hist = np.bincount(pitches, minlength=128)
    c_indices = list(range(0, 128, 12))
    c_count = sum(pitch_hist[i] for i in c_indices)
    features.append("CMajor" if c_count > len(pitches) * 0.25 else "OtherKey")

    return list(set(features))


# ──────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────

def load_inference_model(checkpoint_path, mode_override=None, device="cuda",
                         vqvae_path=None, d_vq=None):
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")

    config_dict = state.get("config", {})
    acfg = AdapterConfig.from_dict(config_dict) if config_dict else AdapterConfig()

    if mode_override:
        acfg.projection_mode = mode_override
    elif not config_dict:
        fname = os.path.basename(checkpoint_path).lower()
        if "vqvae" in fname:
            acfg.projection_mode = "vqvae"
        elif "direct" in fname:
            acfg.projection_mode = "direct"

    if vqvae_path:
        acfg.vqvae_checkpoint = vqvae_path
    if d_vq:
        acfg.d_vq = d_vq

    adapter = AdapterFactory.build(acfg).to(device)
    adapter.load_state_dict(state["adapter"])
    adapter.eval()

    print(f"Loading LLM: {acfg.llm_model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(acfg.llm_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        acfg.llm_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=acfg.qlora_r,
        lora_alpha=acfg.qlora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    llm = get_peft_model(llm, lora_config)

    if "lora" in state:
        llm.load_state_dict(state["lora"], strict=False)
        print("✅ LoRA weights loaded.")

    llm.eval()
    return tokenizer, llm, adapter, acfg


# ──────────────────────────────────────────────
# 主推理函式
# ──────────────────────────────────────────────

def run_inference_with_graph_rag(
    midi_path: str,
    tokenizer,
    llm,
    adapter,
    acfg,
    graph_rag: MusicKnowledgeGraph = None,
    cag_ctx: CAGKV = None,
    device: str = "cuda",
    custom_seq_len: int = None,
):
    """
    完整推理流程：
    1. 提取 OctupleMIDI tokens
    2. 偵測 MIDI 特徵
    3. [可選] CAG：從磁碟載入 KV Cache（預計算文件已嵌入 attention）
    4. [可選] Graph RAG：組成 query 文字部分的上下文
    5. 編碼 query + 拼接 music prefix → generate
    """
    print(f"\nProcessing: {midi_path}")
    effective_seq_len = custom_seq_len or acfg.seq_len

    # ── Step 1: 提取 tokens ────────────────────
    extractor = get_extractor("octuple_8d")
    tokens = extractor.extract(midi_path)
    if tokens is None or len(tokens) == 0:
        raise ValueError("MIDI file yielded no tokens.")

    tokens_to_use = tokens[:effective_seq_len]
    x = torch.from_numpy(tokens_to_use).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():

        # ── Step 2: Adapter 投影 ───────────────
        music_prefix = adapter(x)  # (1, N/4, 2048)
        print(f"   Music prefix: {music_prefix.size()} tokens")

        # ── Step 3: MIDI 特徵偵測 ──────────────
        midi_features = detect_midi_features(tokens_to_use)
        print(f"   Detected features: {', '.join(midi_features)}")

        # ── Step 4: 組裝 query 文字（Graph RAG 部分）──
        query_parts = []

        if graph_rag is not None:
            print("Graph RAG: 查詢知識圖谱（快取命中時跳過網路搜尋）...")
            graph_context = graph_rag.get_analysis_context(midi_features)
            query_parts.append(graph_context)
            print(f"Graph RAG 上下文長度: {len(graph_context)} chars")

        base_prompt = (
            "Based on the knowledge context above, analyze the musical structure, "
            "style, instrumentation, and musical characteristics of this MIDI piece "
            "in detail. Reference specific musical concepts where applicable."
        )
        query_parts.append(base_prompt)
        query_text = "\n\n".join(query_parts)

        # ── Step 5: 編碼 query ─────────────────
        query_inputs = tokenizer(query_text, return_tensors="pt").to(device)
        query_token_len = query_inputs.input_ids.shape[1]
        print(f"Query token 數：{query_token_len}")

        # ── Step 6: 載入 CAG KV Cache ──────────
        past_kv = None
        doc_token_len = 0
        if cag_ctx is not None:
            print("CAG: 載入 KV Cache...")
            past_kv, doc_token_len = cag_ctx.load(
                midi_features, llm, query_token_len=query_token_len
            )
            if past_kv is not None:
                print(f"   KV Cache 已就緒，文件 {doc_token_len} tokens 已預嵌入 attention")
            else:
                print("   KV Cache 未命中，繼續無 CAG 推理")

        # ── Step 7: 拼接 embeds ────────────────
        query_embeds = llm.get_input_embeddings()(query_inputs.input_ids)
        llm_dtype = next(llm.parameters()).dtype

        full_embeds = torch.cat([
            music_prefix.float().to(llm_dtype),
            query_embeds.to(llm_dtype),
        ], dim=1)
        print(f"Total input tokens: {full_embeds.size(1)}  "
              f"(music prefix + query"
              + (f" + {doc_token_len} doc tokens via KV Cache)" if past_kv is not None else ")"))

        # ── Step 8: 生成 ───────────────────────
        print("Generating analysis...")
        generate_kwargs = dict(
            inputs_embeds=full_embeds,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
        # 有 KV Cache 就傳入，LLM 直接在文件 attention 基礎上繼續
        if past_kv is not None:
            generate_kwargs["past_key_values"] = past_kv

        output_ids = llm.generate(**generate_kwargs)

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIDI Analysis with Graph RAG + PDF CAG")

    # 基本參數
    parser.add_argument("--midi",       type=str, required=True,  help="MIDI 文件路徑")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Adapter checkpoint 路徑")
    parser.add_argument("--mode",       choices=["direct", "vqvae"],  help="Projection mode")
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--seq_len",    type=int, help="Sequence length")
    parser.add_argument("--vqvae",      type=str, help="VQ-VAE checkpoint 路徑")
    parser.add_argument("--d_vq",       type=int, help="VQ-VAE hidden dim")

    # Graph RAG 參數
    parser.add_argument(
        "--enable-graph-rag",
        action="store_true",
        help="啟用 Graph RAG（從 Wikipedia 即時搜尋）"
    )
    parser.add_argument(
        "--graph-cache-dir",
        type=str,
        default="graph_cache",
        help="Graph RAG 序列化快取資料夾（預設: graph_cache）"
    )
    parser.add_argument(
        "--export-graph",
        action="store_true",
        help="推理後將知識圖谱匯出為 JSON"
    )

    # CAG 參數
    parser.add_argument(
        "--enable-cag",
        action="store_true",
        help="啟用 CAG（KV Cache 版，從 CAG_source/ PDF 知識庫預計算 KV Cache）"
    )
    parser.add_argument(
        "--cag-source",
        type=str,
        default="CAG_source",
        help="CAG PDF 資料夾路徑（預設: CAG_source）"
    )
    parser.add_argument(
        "--cag-cache-dir",
        type=str,
        default="kv_cache",
        help="KV Cache 存放資料夾（預設: kv_cache）"
    )
    parser.add_argument(
        "--cag-top-k",
        type=int,
        default=3,
        help="CAG 最多選取幾份 PDF（預設: 3）"
    )
    parser.add_argument(
        "--cag-precompute",
        action="store_true",
        help="執行 KV Cache 預計算（第一次或更新文件時使用）"
    )
    parser.add_argument(
        "--cag-force",
        action="store_true",
        help="強制重新預計算 KV Cache（即使已有快取）"
    )

    args = parser.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        # 載入模型
        tokenizer, llm, adapter, config = load_inference_model(
            args.checkpoint, args.mode, args.device,
            vqvae_path=args.vqvae, d_vq=args.d_vq
        )

        # 初始化 Graph RAG（只傳物件進去，實際搜尋在推理時才執行）
        graph_rag = MusicKnowledgeGraph(cache_dir=args.graph_cache_dir) if args.enable_graph_rag else None
        if graph_rag:
            print(f"🔧 Graph RAG 已啟用（快取資料夾: {args.graph_cache_dir}）")

        # 初始化 CAG（KV Cache 版）
        cag_ctx = None
        if args.enable_cag:
            print(f"CAG 已啟用，建立 PDF 索引: {args.cag_source}")
            cag_ctx = CAGKV(
                source_dir=args.cag_source,
                cache_dir=args.cag_cache_dir,
                top_k=args.cag_top_k,
                max_seq_len=acfg.seq_len * 2,  # 估算上限
            )
            print(cag_ctx.list_all_documents())
            print("已快取的 KV Cache：")
            print(cag_ctx.list_cached())

            if args.cag_precompute:
                # 離線預計算模式：先跑 MIDI 特徵偵測，再預計算 KV Cache
                print("\n=== CAG 預計算模式 ===")
                from models.octuple import get_extractor as _get_extractor
                import numpy as np
                _extractor = _get_extractor("octuple_8d")
                _tokens = _extractor.extract(args.midi)
                if _tokens is not None and len(_tokens) > 0:
                    _features = detect_midi_features(_tokens[:acfg.seq_len])
                    print(f"偵測到特徵：{_features}")
                    cag_ctx.precompute(tokenizer, llm, _features, force=args.cag_force)
                else:
                    print("⚠️  MIDI 特徵偵測失敗，跳過預計算")

        # 執行推理
        analysis = run_inference_with_graph_rag(
            midi_path=args.midi,
            tokenizer=tokenizer,
            llm=llm,
            adapter=adapter,
            acfg=config,
            graph_rag=graph_rag,
            cag_ctx=cag_ctx,
            device=args.device,
            custom_seq_len=args.seq_len,
        )

        # 可選：匯出知識圖谱
        if args.export_graph and graph_rag is not None:
            graph_rag.to_json("knowledge_graph.json")

        print("\n" + "=" * 70)
        print("GENERATED ANALYSIS:")
        print("=" * 70)
        print(analysis)
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()