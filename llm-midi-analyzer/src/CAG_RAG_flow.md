# MIDI 分析系統：Graph RAG + CAG 完整流程說明

## 系統架構總覽

本系統結合兩套知識增強機制，在 LLM 生成 MIDI 分析前注入專業音樂知識：

- **Graph RAG**：即時從 Wikipedia 動態建立知識圖譜，作為 query 的文字上下文
- **CAG（KV Cache 版）**：將 PDF 教材預先編碼為 KV Cache，推理時直接注入 attention，不重算文件部分

```
MIDI 檔案
   │
   ├─► [OctupleMIDI 提取] ─► tokens (T, 8)
   │
   ├─► [特徵偵測] ─► midi_features
   │                    │
   │         ┌──────────┴──────────┐
   │         ▼                     ▼
   │   [Graph RAG]            [CAG / KV Cache]
   │   Wikipedia 圖譜          PDF KV Cache
   │   → query 文字上下文      → past_key_values
   │         │                     │
   │         └──────────┬──────────┘
   │                    ▼
   ├─► [Adapter 投影] ─► music_prefix embeds
   │
   └─► [LLM 生成] ─► 分析文字
```

---

## 一、前置：MIDI 特徵偵測

**檔案**：`generate_analysis_v3.py` → `detect_midi_features()`

OctupleMIDI token 的 shape 為 `(T, 8)`，8 個維度依序為：
`[Bar, Position, Program, Pitch, Duration, Velocity, TimeSig, Tempo]`

| 特徵維度 | 判斷邏輯 | 輸出標籤 |
|---|---|---|
| 聲部織體 | 不重複 Program 數 ≥ 3 | `Polyphony` / `Homophony` |
| 主要樂器 | Program 中位數 ≤ 3 → 鍵盤；40–43 → 弦樂 | `Keyboard` / `Strings` / `WindInstruments` |
| 速度 | Tempo bucket 換算 BPM | `FastTempo` / `ModerateTempo` / `SlowTempo` |
| 形式重複性 | 前後半段 Pitch 平均差 < 5 | `RepetitiveForm` / `DevelopingForm` |
| 調性 | C 音（含各八度）佔比 > 25% | `CMajor` / `OtherKey` |

偵測結果（例如 `["Polyphony", "Keyboard", "ModerateTempo", "RepetitiveForm", "CMajor"]`）是後續兩套機制的共同輸入。

---

## 二、Graph RAG 流程

**檔案**：`graph_rag.py` → `MusicKnowledgeGraph`

### 2.1 索引期（第一次或快取未命中）

```
midi_features
   │
   ▼
FEATURE_TO_CONCEPTS 映射表
   │  e.g. "Polyphony" → [("Polyphony", "technique"), ("Counterpoint", "technique"), ("Fugue", "form")]
   ▼
種子概念列表（去重）
   │
   ├─► Wikipedia API 查詢（每個種子概念）
   │     → 摘要文字（前 600 字）+ 類別標籤
   │     → 加入 nx.DiGraph 作為節點
   │
   ├─► 擴展一層：取每個種子概念的音樂相關連結（max 4）
   │     → 新概念節點 + related_to 邊
   │
   ├─► EntityRelationExtractor：從摘要文字正則抽取三元組
   │     e.g. "Fugue is a form of counterpoint" → (Fugue, is_form_of, counterpoint)
   │     → 新增邊（confidence=0.5）
   │
   ├─► CommunityDetector：greedy_modularity_communities
   │     → 高度連結子圖分群
   │     → 每群生成摘要（度數最高的前 3 個節點）
   │
   └─► 序列化存檔 → graph_cache/graph_<features_hash>.json
```

### 2.2 查詢期（快取命中）

```
graph_<features_hash>.json
   │
   └─► 還原 nx.DiGraph + communities + community_summaries
       （跳過所有 Wikipedia 網路請求）
```

### 2.3 格式化為 query 上下文

`_format_context()` 依序輸出：

1. 各類型節點列表（演奏技術 / 音樂形式 / 樂器 / 樂理概念 / 延伸概念）
2. 核心概念摘要（前 6 個種子概念的 Wikipedia 第一段，截 300 字）
3. 概念關聯邊（最多 10 條，格式：`Fugue → Counterpoint [is_form_of]`）
4. Query-Focused 聚合：`QueryFocusedAggregator` 根據 MIDI 特徵對社群排名，取 top-3 社群摘要

最終輸出為純文字字串，拼入 prompt 的 query 部分。

---

## 三、CAG 流程（KV Cache 版）

**檔案**：`cag.py` → `CAGKV`

### 3.1 外部知識預加載（離線，只跑一次）

```
midi_features
   │
   ▼
CAGSourceIndex.retrieve()
   │  Jaccard-like 關鍵字評分 + 分類加權 + 檔名加權
   │  → 選出 top_k 份 PDF（預設 3 份）
   ▼
PDFExtractor.extract()
   │  pypdf 讀取，最多 1500 chars/份
   ▼
doc_text（拼接所有 PDF 內容，分隔符 "---"）
   │
   ▼
tokenizer（truncation, max_length = max_seq_len // 2）
   │  文件最多佔一半 context window
   ▼
LLM forward(use_cache=True)
   │  不生成 token，只做 prefill
   ▼
past_key_values
   │  每層 (key, value) tensor，shape = (1, heads, doc_seq, head_dim)
   ▼
_kv_to_cpu()  所有 tensor 搬到 CPU
   │
   ▼
torch.save() → kv_cache/kvcache_<features>_<md5>.pt
```

> **快取 key** 由 `sorted(midi_features)` join 後取 MD5 前 8 碼決定，特徵順序不影響命中。

### 3.2 推理時載入

```
kv_cache/kvcache_<features>_<md5>.pt
   │
   ▼
torch.load(map_location="cpu")
   │
   ▼
_kv_to_device()
   │  利用 llm.model.layers[i] 的參數 device
   │  將各層 KV tensor 搬回對應 GPU（解決量化模型跨 device 問題）
   ▼
Truncate 檢查
   │  if doc_token_len + query_token_len > max_seq_len:
   │      keep = max_seq_len - query_token_len
   │      k = k[:, :, -keep:, :]  ← 保留最後 keep 個 seq 位置
   ▼
past_key_values（就緒）
```

### 3.3 注入推理

```
past_key_values（文件 attention 已預計算）
   +
inputs_embeds（music_prefix + query embeds 拼接）
   │
   ▼
llm.generate(inputs_embeds=..., past_key_values=...)
   │
   │  LLM 直接在文件 KV Cache 基礎上繼續 attention
   │  文件部分不重新計算，節省大量計算量
   ▼
生成 token → 分析文字
```

---

## 四、推理完整流程（含兩套機制合併）

**檔案**：`generate_analysis_v3.py` → `run_inference_with_graph_rag()`

```
Step 1  OctupleMIDI 提取
        midi_path → tokens (T, 8)

Step 2  Adapter 投影
        tokens → Adapter → music_prefix (1, N/4, 2048)

Step 3  MIDI 特徵偵測
        tokens → detect_midi_features() → midi_features

Step 4  Graph RAG（可選）
        midi_features → MusicKnowledgeGraph.get_analysis_context()
                      → graph_context 字串（加入 query_parts）

Step 5  組合 query 文字
        query_parts + base_prompt → query_text
        tokenizer(query_text) → query_token_len

Step 6  CAG KV Cache 載入（可選）
        midi_features + query_token_len → CAGKV.load()
                                        → past_kv, doc_token_len
                                        （含 Truncate 處理）

Step 7  拼接 embeds
        music_prefix  ← Adapter 輸出
        query_embeds  ← tokenizer → embedding layer
        full_embeds   = cat([music_prefix, query_embeds], dim=1)

Step 8  LLM 生成
        llm.generate(
            inputs_embeds = full_embeds,
            past_key_values = past_kv,   ← 文件知識已嵌入
            max_new_tokens = 256,
            ...
        ) → response
```

### Context Window 組成

```
┌─────────────────────────────────────────────────────┐
│  past_key_values（CAG）                              │
│  └── PDF 文件內容，已預計算，不占 inputs_embeds      │
│       doc_token_len（最多 max_seq_len // 2）         │
├─────────────────────────────────────────────────────┤
│  inputs_embeds                                       │
│  ├── music_prefix  (1, N/4, 2048)                   │
│  │    └── MIDI token 經 Adapter 投影                 │
│  └── query_embeds                                    │
│       ├── [Graph RAG context]  Wikipedia 知識圖譜    │
│       └── base_prompt          分析指令              │
└─────────────────────────────────────────────────────┘
```

---

## 五、快取策略彙整

| 機制 | 快取格式 | 快取 key | 命中效果 |
|---|---|---|---|
| Graph RAG | `graph_cache/*.json` | sorted features MD5 | 跳過所有 Wikipedia 網路請求 |
| CAG | `kv_cache/*.pt` | sorted features MD5 | 跳過 LLM prefill，直接載入 KV |

---

## 六、CLI 使用方式

```bash
# 第一次：預計算 CAG KV Cache
python generate_analysis_v3.py \
    --midi piece.mid \
    --checkpoint ckpt.pt \
    --enable-cag \
    --cag-precompute \
    --cag-cache-dir kv_cache

# 正式推理（Graph RAG + CAG）
python generate_analysis_v3.py \
    --midi piece.mid \
    --checkpoint ckpt.pt \
    --enable-graph-rag \
    --graph-cache-dir graph_cache \
    --enable-cag \
    --cag-cache-dir kv_cache

# 更新 PDF 後強制重算 KV Cache
python generate_analysis_v3.py \
    --midi piece.mid \
    --checkpoint ckpt.pt \
    --enable-cag \
    --cag-precompute \
    --cag-force

# 匯出知識圖譜 JSON（除錯用）
python generate_analysis_v3.py \
    --midi piece.mid \
    --checkpoint ckpt.pt \
    --enable-graph-rag \
    --export-graph
```

---

## 七、兩套機制比較

| | Graph RAG | CAG（KV Cache）|
|---|---|---|
| 知識來源 | Wikipedia（動態） | 精選 PDF 教材（靜態）|
| 注入方式 | 文字拼入 query prompt | KV Cache 注入 attention 層 |
| 推理時計算量 | 文字部分需重算 attention | 文件部分完全跳過 |
| 知識結構 | 概念圖（節點 + 關係邊）| 純文字段落 |
| 領域深度 | Wikipedia 摘要級別 | 專業教材級別 |
| 網路依賴 | 是（第一次建圖）| 否 |
| 快取粒度 | 特徵組合 → JSON | 特徵組合 → `.pt` |