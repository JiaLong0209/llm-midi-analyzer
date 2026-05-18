"""
cag.py — Cache-Augmented Generation (真正的 KV Cache 版本)
============================================================
架構對應圖中流程：
  外部知識預加載：
    document set → LLM 編碼 → precomputed KV Cache → 存磁碟
  推理時：
    從磁碟載入 KV Cache → 內存 → 與 Query 拼接 → LLM → resp

  Truncate / Cache 重置：
    當 KV Cache + Query 超過 max_seq_len，自動截斷舊 Cache。

使用方式：
    cag = CAGKV(source_dir="CAG_source", cache_dir="kv_cache")
    cag.precompute(tokenizer, llm, midi_features)   # 離線預計算（只需跑一次）
    past_kv, doc_len = cag.load(midi_features, llm) # 推理時載入
"""

import os
import re
import torch
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pypdf import PdfReader


# ──────────────────────────────────────────────
# 1. PDF 文字提取
# ──────────────────────────────────────────────

class PDFExtractor:
    """從 PDF 提取純文字（帶記憶體快取）"""

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def extract(self, pdf_path: str, max_chars: int = 3000) -> str:
        if pdf_path in self._cache:
            return self._cache[pdf_path]

        if not os.path.exists(pdf_path):
            return f"[文件不存在: {pdf_path}]"

        try:
            reader = PdfReader(pdf_path)
            texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
            full_text = re.sub(r"\n{3,}", "\n\n", "\n".join(texts))[:max_chars]
            self._cache[pdf_path] = full_text
            return full_text
        except Exception as e:
            msg = f"[PDF 讀取失敗: {e}]"
            self._cache[pdf_path] = msg
            return msg


# ──────────────────────────────────────────────
# 2. PDF 索引（選出相關文件）
# ──────────────────────────────────────────────

@dataclass
class PDFDocument:
    path: str
    filename: str
    category: str
    keywords: List[str] = field(default_factory=list)


class CAGSourceIndex:
    CATEGORY_ALIAS: Dict[str, str] = {
        "配器法": "orchestration",
        "對位法": "counterpoint",
        "和聲學": "harmony",
        "基礎樂理": "basic_theory",
        "曲式學": "musical_form",
        "其他":   "other",
    }

    FEATURE_KEYWORD_MAP: Dict[str, List[str]] = {
        "Polyphony":       ["counterpoint", "musical", "form", "basic"],
        "Homophony":       ["harmony", "basic", "theory"],
        "Keyboard":        ["orchestration", "basic", "theory"],
        "Strings":         ["orchestration"],
        "WindInstruments": ["orchestration"],
        "FastTempo":       ["basic", "theory", "musical", "form"],
        "SlowTempo":       ["basic", "theory"],
        "ModerateTempo":   ["basic", "theory"],
        "RepetitiveForm":  ["musical", "form"],
        "DevelopingForm":  ["musical", "form", "counterpoint"],
        "CMajor":          ["harmony", "basic", "theory"],
        "OtherKey":        ["harmony", "basic", "theory"],
        "Unknown":         ["basic", "theory"],
    }

    def __init__(self, source_dir: str = "CAG_source"):
        self.source_dir = Path(source_dir)
        self.documents: List[PDFDocument] = []
        self._build_index()

    def _build_index(self):
        pdf_files = list(self.source_dir.rglob("*.pdf"))
        print(f"CAG 索引：找到 {len(pdf_files)} 個 PDF 文件")

        for pdf_path in pdf_files:
            try:
                rel = pdf_path.relative_to(self.source_dir)
            except ValueError:
                rel = pdf_path

            parts = rel.parts
            category = parts[0] if len(parts) > 1 else "其他"
            filename = pdf_path.stem
            category_en = self.CATEGORY_ALIAS.get(category, category.lower())
            raw_kws = category_en.split("_") + re.split(r"[_\-\s]+", filename.lower())
            keywords = [k for k in raw_kws if len(k) > 1]

            self.documents.append(PDFDocument(
                path=str(pdf_path),
                filename=filename,
                category=category,
                keywords=keywords,
            ))

        if self.documents:
            print(f"   分類: {', '.join(sorted({d.category for d in self.documents}))}")
        else:
            print("未找到任何 PDF，請確認 CAG_source 路徑正確")

    def retrieve(self, midi_features: List[str], top_k: int = 3) -> List[Tuple[PDFDocument, float]]:
        if not self.documents:
            return []

        query_kws = set(k.lower() for k in self._features_to_keywords(midi_features))
        scored = []
        for doc in self.documents:
            doc_kws = set(doc.keywords)
            inter = query_kws & doc_kws
            score = len(inter) / max(len(query_kws | doc_kws), 1) if inter else 0.0
            cat_en = self.CATEGORY_ALIAS.get(doc.category, doc.category.lower())
            if cat_en in query_kws or any(k in cat_en for k in query_kws):
                score += 0.4
            if any(k in doc.filename.lower() for k in query_kws):
                score += 0.2
            scored.append((doc, round(score, 4)))

        scored.sort(key=lambda x: x[1], reverse=True)
        result = scored[:top_k]
        if all(s == 0 for _, s in result):
            print("CAG：找不到高度相關文件，使用兜底文件")
        return result

    def _features_to_keywords(self, features: List[str]) -> List[str]:
        kws = []
        for f in features:
            if f in self.FEATURE_KEYWORD_MAP:
                kws.extend(self.FEATURE_KEYWORD_MAP[f])
            else:
                # Split complex phrases into individual words (e.g. "I -> IV" -> ["i", "iv"])
                # Also filter out very short strings like "->"
                sub_kws = re.split(r"[^a-zA-Z0-9]+", f.lower())
                kws.extend([k for k in sub_kws if len(k) > 1])
        return kws


# ──────────────────────────────────────────────
# 3. KV Cache 序列化工具
# ──────────────────────────────────────────────

def _kv_to_cpu(past_key_values) -> List[List[torch.Tensor]]:
    """所有 tensor 搬到 CPU，供序列化用"""
    return [[k.cpu(), v.cpu()] for k, v in past_key_values]


def _kv_to_device(cpu_kv: List[List[torch.Tensor]], llm) -> tuple:
    """
    將 CPU KV Cache 搬回各 layer 對應的 device。
    利用 llm.model.layers[i] 的參數 device 決定目標。
    """
    result = []
    for layer_idx, (k, v) in enumerate(cpu_kv):
        try:
            layer = llm.model.layers[layer_idx]
            target = next(layer.parameters()).device
        except (AttributeError, StopIteration, IndexError):
            target = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        result.append((k.to(target), v.to(target)))
    return tuple(result)


def _save_kv(past_key_values, cache_path: Path, doc_token_len: int):
    payload = {
        "doc_token_len": doc_token_len,
        "num_layers": len(past_key_values),
        "kv": _kv_to_cpu(past_key_values),
    }
    torch.save(payload, cache_path)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"KV Cache 存檔 → {cache_path.name}  ({size_mb:.1f} MB, {doc_token_len} doc tokens)")


def _load_kv(cache_path: Path, llm) -> Tuple[tuple, int]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    past_kv = _kv_to_device(payload["kv"], llm)
    doc_len = payload["doc_token_len"]
    size_mb = cache_path.stat().st_size / 1024 / 1024
    print(f"✅ KV Cache 載入：{cache_path.name}  "
          f"({payload['num_layers']} layers, {doc_len} doc tokens, {size_mb:.1f} MB)")
    return past_kv, doc_len


# ──────────────────────────────────────────────
# 4. 主類：CAGKV
# ──────────────────────────────────────────────

class CAGKV:
    """
    Cache-Augmented Generation（KV Cache 版）

    外部知識預加載（離線）：
        cag.precompute(tokenizer, llm, midi_features)
        → PDF 文字 → tokenize → LLM forward（use_cache=True）
        → past_key_values → 磁碟

    推理時：
        past_kv, doc_len = cag.load(midi_features, llm, query_token_len)
        → 磁碟 → KV Cache（自動 Truncate 若超過 max_seq_len）
        → 傳入 llm.generate(past_key_values=past_kv)
    """

    def __init__(
        self,
        source_dir: str = "CAG_source",
        cache_dir: str = "kv_cache",
        top_k: int = 3,
        max_chars_per_doc: int = 1500,
        max_seq_len: int = 4096,
    ):
        self.index = CAGSourceIndex(source_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.max_chars_per_doc = max_chars_per_doc
        self.max_seq_len = max_seq_len
        self._extractor = PDFExtractor()

    # ── 快取 key ─────────────────────────────────
    def _cache_key(self, midi_features: List[str]) -> str:
        key = "_".join(sorted(midi_features))
        h = hashlib.md5(key.encode()).hexdigest()[:8]
        safe = re.sub(r"[^\w]", "-", key)[:60]
        return f"kvcache_{safe}_{h}.pt"

    def _cache_path(self, midi_features: List[str]) -> Path:
        return self.cache_dir / self._cache_key(midi_features)

    # ── 組合文件文字 ──────────────────────────────
    def _build_doc_text(self, midi_features: List[str]) -> str:
        from tqdm import tqdm
        retrieved = self.index.retrieve(midi_features, top_k=self.top_k)
        if not retrieved:
            return ""
        parts = []
        for doc, _ in tqdm(retrieved, desc="   -> 正在從 PDF 教科書提取相關段落"):
            text = self._extractor.extract(doc.path, max_chars=self.max_chars_per_doc)
            parts.append(f"[{doc.category} / {doc.filename}]\n{text}")
        return "\n\n---\n\n".join(parts)

    # ── 離線預計算 ────────────────────────────────
    def precompute(
        self,
        tokenizer,
        llm,
        midi_features: List[str],
        force: bool = False,
    ) -> Path:
        """
        把選出的 PDF 文字喂給 LLM，預計算 KV Cache 存磁碟。

        參數：
            force=True  強制重算，忽略已有快取
        回傳：
            快取檔路徑
        """
        cache_path = self._cache_path(midi_features)

        if cache_path.exists() and not force:
            print(f"KV Cache 已存在，跳過預計算：{cache_path.name}")
            return cache_path

        doc_text = self._build_doc_text(midi_features)
        if not doc_text:
            print("CAG precompute: 沒有找到相關文件，略過")
            return cache_path

        print(f"CAG 預計算：文件 {len(doc_text)} chars → tokenize...")
        # 文件最多佔一半 context window，留另一半給 query + music prefix
        inputs = tokenizer(
            doc_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len // 2,
        )
        doc_token_len = inputs.input_ids.shape[1]
        print(f"   文件 token 數：{doc_token_len}")

        # 找第一個參數的 device 作為輸入 device
        try:
            first_device = next(llm.parameters()).device
        except StopIteration:
            first_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_ids = inputs.input_ids.to(first_device)

        print("   LLM forward（預計算 KV Cache）...")
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = llm(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        _save_kv(outputs.past_key_values, cache_path, doc_token_len)
        return cache_path

    # ── 推理時載入 ────────────────────────────────
    def load(
        self,
        midi_features: List[str],
        llm,
        query_token_len: int = 0,
    ) -> Tuple[Optional[tuple], int]:
        """
        從磁碟載入 KV Cache，自動 Truncate。

        參數：
            query_token_len  已知的 query token 數，用於 Truncate 判斷
        回傳：
            (past_key_values, doc_token_len)
            快取不存在時回傳 (None, 0)
        """
        cache_path = self._cache_path(midi_features)
        if not cache_path.exists():
            print("CAG load: 找不到 KV Cache，建議先執行 precompute()")
            return None, 0

        past_kv, doc_token_len = _load_kv(cache_path, llm)

        # ── Truncate 邏輯（對應圖中 Cache 重置）─────
        total = doc_token_len + query_token_len
        if total > self.max_seq_len:
            keep = max(self.max_seq_len - query_token_len, 0)
            print(f"⚠️  KV Truncate：doc {doc_token_len} → {keep} tokens "
                  f"（{total} 超過上限 {self.max_seq_len}）")
            # 沿 seq 維度（dim=2）保留最後 keep 個位置
            truncated = []
            for k, v in past_kv:
                k_t = k[:, :, -keep:, :] if keep > 0 else k[:, :, :0, :]
                v_t = v[:, :, -keep:, :] if keep > 0 else v[:, :, :0, :]
                truncated.append((k_t, v_t))
            past_kv = tuple(truncated)
            doc_token_len = keep

        return past_kv, doc_token_len

    # ── 除錯工具 ──────────────────────────────────
    def list_cached(self) -> str:
        files = sorted(self.cache_dir.glob("kvcache_*.pt"))
        if not files:
            return "（尚無 KV Cache）"
        lines = [f"  {f.name}  ({f.stat().st_size/1024/1024:.1f} MB)" for f in files]
        return "\n".join(lines)

    def list_all_documents(self) -> str:
        if not self.index.documents:
            return "（尚無文件）"
        return "\n".join(f"  [{d.category}] {d.filename}.pdf" for d in self.index.documents)