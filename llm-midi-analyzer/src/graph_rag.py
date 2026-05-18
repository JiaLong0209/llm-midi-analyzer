"""
graph_rag.py - Graph RAG 實現（純網路搜尋版）
================================================
不使用任何預設靜態圖。
完全根據 MIDI 特徵即時從網路（Wikipedia）查詢，
動態建立臨時知識圖谱，供 LLM 分析使用。
"""

import networkx as nx
import requests
import json
import re
import os
import time
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict


# ──────────────────────────────────────────────
# 1. Wikipedia 搜尋工具
# ──────────────────────────────────────────────

class WikipediaSearcher:
    """輕量 Wikipedia API 封裝"""

    BASE_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS  = {"User-Agent": "MusicGraphRAG/1.0 (research project)"}

    def __init__(self, timeout: int = 6, lang: str = "zh-tw"):
        self.timeout = timeout
        self._cache: Dict[str, Optional[Dict]] = {}
        self.retry_attempts = 3
        self.retry_delay = 0.5
        
        # Subdomain selection based on lang parameter
        subdomain = "zh"
        if not lang:
            subdomain = "zh"
        else:
            lang_lower = lang.lower().strip()
            if "zh-tw" in lang_lower or "traditional chinese" in lang_lower or "繁體" in lang_lower:
                subdomain = "zh"
            elif lang_lower in ["en", "english"]:
                subdomain = "en"
            elif lang_lower in ["ja", "japanese", "日本語"]:
                subdomain = "ja"
            elif lang_lower in ["ko", "korean", "한국어"]:
                subdomain = "ko"
            elif lang_lower in ["ru", "russian", "русский"]:
                subdomain = "ru"
            else:
                # Custom input (e.g. "es", "de", "fr")
                # Fallback to standard 2-letter code if provided, or defaults to "zh"
                matched = re.match(r'^([a-z]{2})', lang_lower)
                if matched:
                    subdomain = matched.group(1)
                else:
                    subdomain = "zh"
                    
        self.BASE_URL = f"https://{subdomain}.wikipedia.org/w/api.php"
        print(f"   [WikipediaSearcher] Using language: {subdomain} -> BASE_URL: {self.BASE_URL}")

    # ── 全文摘要 ──────────────────────────────
    def fetch_summary(self, concept: str) -> Optional[Dict]:
        """
        取得 Wikipedia 摘要（前 600 字）。
        回傳 {"title": ..., "extract": ..., "categories": [...]}
        """
        if concept in self._cache:
            return self._cache[concept]

        result = self._query_page(concept)
        self._cache[concept] = result
        return result

    # ── 相關概念（連結） ──────────────────────
    def fetch_related_concepts(self, concept: str, max_links: int = 8) -> List[str]:
        """從 Wikipedia 頁面的連結中提取「音樂相關」的概念。"""
        for attempt in range(self.retry_attempts):
            try:
                params = {
                    "action": "query",
                    "titles": concept,
                    "prop": "links",
                    "pllimit": 50,
                    "plnamespace": 0,
                    "format": "json",
                }
                resp = requests.get(
                    self.BASE_URL, params=params,
                    headers=self.HEADERS, timeout=self.timeout
                )
                
                if resp.status_code != 200:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
                try:
                    data = resp.json()
                except ValueError:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                pages = data.get("query", {}).get("pages", {})

                links = []
                for page_data in pages.values():
                    for link in page_data.get("links", []):
                        title = link.get("title", "")
                        if self._is_music_related(title):
                            links.append(title)
                            if len(links) >= max_links:
                                break

                return links

            except Exception as e:
                print(f"   ⚠️  fetch_related_concepts('{concept}') failed (Attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay * (attempt + 1))

        return []

    # ── 搜尋建議（處理拼字變體） ─────────────
    def search_suggestions(self, query: str, limit: int = 3) -> List[str]:
        """當精確標題找不到時，改用搜尋建議"""
        try:
            params = {
                "action": "opensearch",
                "search": query,
                "limit": limit,
                "format": "json",
            }
            resp = requests.get(
                self.BASE_URL, params=params,
                headers=self.HEADERS, timeout=self.timeout
            )
            data = resp.json()
            return data[1] if len(data) > 1 else []
        except Exception:
            return []

    # ── 私有：查 Wikipedia 頁面 ───────────────
    def _query_page(self, title: str) -> Optional[Dict]:
        for attempt in range(self.retry_attempts):
            try:
                params = {
                    "action": "query",
                    "titles": title,
                    "prop": "extracts|categories",
                    "exintro": True,
                    "explaintext": True,
                    "cllimit": 10,
                    "format": "json",
                }
                resp = requests.get(
                    self.BASE_URL, params=params,
                    headers=self.HEADERS, timeout=self.timeout
                )
                
                if resp.status_code != 200:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                # 檢查是否為 JSON
                try:
                    data = resp.json()
                except ValueError:
                    print(f"   ⚠️  Wikipedia returned non-JSON response for '{title}' (Attempt {attempt+1})")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue

                pages = data.get("query", {}).get("pages", {})

                for pid, page in pages.items():
                    if pid == "-1":
                        # 找不到精確標題，試搜尋建議
                        suggestions = self.search_suggestions(title)
                        if suggestions and suggestions[0].lower() != title.lower():
                            return self._query_page(suggestions[0])
                        return None

                    extract = page.get("extract", "")
                    cats = [
                        c["title"].replace("Category:", "")
                        for c in page.get("categories", [])
                    ]
                    return {
                        "title": page.get("title", title),
                        "extract": extract[:2000],
                        "categories": cats,
                        "source": "wikipedia",
                    }
            except Exception as e:
                print(f"   ⚠️  Wikipedia query attempt {attempt+1} failed for '{title}': {e}")
                time.sleep(self.retry_delay * (attempt + 1))

        return None

    # ── 私有：過濾出音樂相關連結 ─────────────
    MUSIC_KEYWORDS = {
        "music", "musical", "melody", "harmony", "rhythm", "tempo",
        "scale", "chord", "key", "mode", "form", "style", "period",
        "baroque", "classical", "romantic", "renaissance", "medieval",
        "fugue", "sonata", "concerto", "suite", "cantata", "oratorio",
        "polyphony", "monophony", "homophony", "counterpoint",
        "piano", "organ", "harpsichord", "violin", "cello", "flute",
        "composer", "bach", "mozart", "beethoven", "handel", "vivaldi",
        "instrument", "ensemble", "orchestra", "quartet",
        "note", "pitch", "interval", "tone", "timbre", "dynamics",
    }

    def _is_music_related(self, title: str) -> bool:
        lower = title.lower()
        return any(kw in lower for kw in self.MUSIC_KEYWORDS)


# ──────────────────────────────────────────────
# 1.5. 實體/關係抽取工具（簡易版）
# ──────────────────────────────────────────────

class EntityRelationExtractor:
    """
    簡易版：從 Wikipedia 文本抽取三元組 (Entity1 - Relation - Entity2)
    使用正則模式匹配（不用 LLM）
    """
    
    # 常見音樂關係模式
    RELATION_PATTERNS = [
        (r"(\w+)\s+(?:is a|is an)\s+(?:form|technique|style)\s+(?:of|for)\s+(\w+)", "is_form_of"),
        (r"(\w+)\s+(?:composed|written|created)\s+(?:by|in)\s+(\w+)", "composed_by"),
        (r"(\w+)\s+originated\s+(?:from|in|during)\s+(\w+)", "originated_from"),
        (r"(\w+)\s+(?:related to|related with)\s+(\w+)", "related_to"),
        (r"(\w+)\s+(?:uses|employs|features)\s+(\w+)", "uses"),
        (r"(\w+)\s+(?:characterized by|known for)\s+(\w+)", "characterized_by"),
    ]
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """從文本抽取三元組列表（增強過濾）。"""
        relations = []
        sentences = re.split(r'[.!?]+', text)
        
        # 垃圾詞黑名單
        blacklist = {
            "which", "first", "free", "this", "there", "from", "with", "they", 
            "each", "many", "some", "often", "used", "also", "then", "when", 
            "work", "part", "song", "compositions", "example", "these", "other"
        }
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            for pattern, relation_type in self.RELATION_PATTERNS:
                matches = re.finditer(pattern, sent, re.IGNORECASE)
                for match in matches:
                    e1, e2 = match.groups()
                    e1, e2 = e1.strip(), e2.strip()
                    
                    # 只有當兩個實體都不在黑名單且長度足夠時才加入
                    if (e1.lower() not in blacklist and e2.lower() not in blacklist and 
                        len(e1) > 2 and len(e2) > 2):
                        relations.append((e1, relation_type, e2))
        
        return relations


# ──────────────────────────────────────────────
# 1.6. 社群偵測與摘要
# ──────────────────────────────────────────────

class CommunityDetector:
    """
    偵測知識圖譜中的社群（高度連接的子圖）
    並為每個社群生成摘要
    """
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph.to_undirected()  # 轉無向圖以進行社群偵測
        self.communities = []
        self.community_summaries = {}
    
    def detect(self) -> List[List[str]]:
        """
        使用貪心模塊化算法偵測社群。
        回傳 [社群1節點列表, 社群2節點列表, ...]
        """
        try:
            from networkx.algorithms import community
            communities = list(community.greedy_modularity_communities(self.graph))
            self.communities = [list(c) for c in communities]
            print(f"社群偵測完成：找到 {len(self.communities)} 個社群")
            return self.communities
        except Exception as e:
            print(f"⚠️  社群偵測失敗: {e}，使用備選方案（按連接度聚類）")
            self._fallback_clustering()
            return self.communities
    
    def _fallback_clustering(self):
        """備選方案：簡單連通分量"""
        try:
            from networkx.algorithms import components
            self.communities = [list(c) for c in components.connected_components(self.graph)]
        except Exception:
            self.communities = [[n] for n in self.graph.nodes()]
    
    def summarize_communities(self, original_graph: nx.DiGraph) -> Dict[int, str]:
        """
        為每個社群生成摘要（簡易版：列出主要節點 + 類型）
        回傳 {社群ID: 摘要文本}
        """
        if not self.communities:
            self.detect()
        
        summaries = {}
        for i, community_nodes in enumerate(self.communities):
            # 選出該社群中的主要節點（度數最高的前 3 個）
            node_degrees = [(n, original_graph.degree(n)) for n in community_nodes if n in original_graph]
            node_degrees.sort(key=lambda x: x[1], reverse=True)
            top_nodes = [n for n, _ in node_degrees[:3]]
            
            # 抽取節點類型
            types = set()
            for n in community_nodes:
                if n in original_graph:
                    ntype = original_graph.nodes[n].get("type", "other")
                    types.add(ntype)
            
            type_str = ", ".join(sorted(types))
            summary = f"[社群 {i}] {', '.join(top_nodes)} ({type_str})"
            summaries[i] = summary
            self.community_summaries[i] = summary
        
        return summaries


# ──────────────────────────────────────────────
# 1.7. 查詢聚合（Query-Focused Aggregation）
# ──────────────────────────────────────────────

class QueryFocusedAggregator:
    """
    根據 MIDI 特徵優先排序社群，聚合相關信息為全局答案
    """
    
    def __init__(self, graph: nx.DiGraph, communities: List[List[str]], 
                 community_summaries: Dict[int, str]):
        self.graph = graph
        self.communities = communities
        self.community_summaries = community_summaries
    
    def rank_communities(self, midi_features: List[str], top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        根據 MIDI 特徵對社群排名。
        回傳 [(社群ID, 相關度分數, 摘要), ...]
        """
        # 簡易相關度：MIDI 特徵中有多少個在社群節點名稱中
        feature_keywords = set()
        for feat in midi_features:
            feature_keywords.add(feat.lower())
            # 分解複合名稱
            feature_keywords.update(re.split(r'(?=[A-Z])', feat.lower()))
        
        scored_communities = []
        for i, nodes in enumerate(self.communities):
            # 計算該社群與特徵的重疊度
            node_str = " ".join(n.lower() for n in nodes)
            overlap = sum(1 for kw in feature_keywords if kw in node_str)
            score = overlap / max(len(feature_keywords), 1)
            
            summary = self.community_summaries.get(i, f"[社群 {i}] {', '.join(nodes[:2])}")
            scored_communities.append((i, score, summary))
        
        # 排序並取 top_k
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        return scored_communities[:top_k]
    
    def aggregate_result(self, midi_features: List[str], top_k: int = 3) -> str:
        """
        聚合最相關的社群，生成全局答案文本
        """
        ranked = self.rank_communities(midi_features, top_k)
        
        lines = ["**Query-Focused 知識聚合**:\n"]
        for comm_id, score, summary in ranked:
            if score > 0:
                lines.append(f"- {summary}  (相關度: {score:.2f})")
            else:
                lines.append(f"- {summary}")
        
        return "\n".join(lines)



class MusicKnowledgeGraph:
    """
    純動態知識圖谱：
    - 不預設任何節點或邊
    - 完全由 MIDI 特徵驅動，即時從 Wikipedia 取資料
    - 每個 MusicKnowledgeGraph 實例對應一次推理
    """
    # MIDI 特徵 → 可搜尋的音樂概念對應表
    FEATURE_TO_CONCEPTS: Dict[str, List[Tuple[str, str]]] = {
        # (Wikipedia 標題, 節點類型)
        "Polyphony":        [("Polyphony", "technique"), ("Counterpoint", "technique"),
                             ("Fugue", "form")],
        "Homophony":        [("Homophony", "technique"), ("Chorale", "form")],
        "Keyboard":         [("Piano", "instrument"), ("Harpsichord", "instrument"),
                             ("Keyboard instrument", "instrument")],
        "Strings":          [("String instrument", "instrument"), ("Violin", "instrument"),
                             ("Cello", "instrument")],
        "WindInstruments":  [("Wind instrument", "instrument"), ("Flute", "instrument"),
                             ("Oboe", "instrument")],
        "FastTempo":        [("Presto (music)", "technique"), ("Virtuosity", "technique")],
        "SlowTempo":        [("Adagio (music)", "technique"), ("Largo", "technique")],
        "ModerateTempo":    [("Andante", "technique")],
        "RepetitiveForm":   [("Musical form", "form"), ("Variation (music)", "form"),
                             ("Rondo", "form")],
        "DevelopingForm":   [("Sonata form", "form"), ("Through-composed", "form")],
        "CMajor":           [("C major", "theory"), ("Diatonic scale", "theory")],
        "OtherKey":         [("Key (music)", "theory"), ("Mode (music)", "theory")],
    }

    def __init__(self, use_web_search: bool = True, search_timeout: int = 6, lang: str = "zh-tw"):
        self.graph = nx.DiGraph()
        self.use_web_search = use_web_search
        self.searcher = WikipediaSearcher(timeout=search_timeout, lang=lang)
        self.extractor = EntityRelationExtractor()
        self.communities = None
        self.community_summaries = None

    # ── 主入口：根據 MIDI 特徵建圖並回傳上下文 ─
    def get_analysis_context(self, midi_features: List[str]) -> str:
        """
        完整流程（含社群偵測與聚合）：
        1. 把 MIDI 特徵展開為初始概念
        2. 查 Wikipedia 取得摘要 + 相關概念
        3. 把相關概念再查一層（深度 = 1）
        4. [新增] 從摘要抽取實體/關係，擴充邊
        5. [新增] 偵測社群、生成社群摘要
        6. [新增] Query-focused 排序與聚合
        7. 格式化為 LLM 提示詞文字
        """
        from tqdm import tqdm
        print(f"Graph RAG: 開始網路搜尋，特徵={midi_features}")

        # Step 1: 展開初始概念
        seed_concepts: List[Tuple[str, str]] = []
        for feat in midi_features:
            if feat in self.FEATURE_TO_CONCEPTS:
                seed_concepts.extend(self.FEATURE_TO_CONCEPTS[feat])
            else:
                # Dynamic keyword from Gemini
                seed_concepts.append((feat, "extracted_entity"))

        # 去重（保留順序）
        seen = set()
        unique_seeds = []
        for title, ntype in seed_concepts:
            if title not in seen:
                seen.add(title)
                unique_seeds.append((title, ntype))

        # Step 2: 查詢種子概念
        for title, ntype in tqdm(unique_seeds, desc="   -> 正在抓取核心概念摘要"):
            self._fetch_and_add_node(title, ntype)
            time.sleep(0.05)

        # Step 3: 從種子概念擴展一層
        expanded = []
        for title, _ in tqdm(unique_seeds, desc="   -> 正在擴展相關知識節點"):
            if title in self.graph:
                related = self.searcher.fetch_related_concepts(title, max_links=4)
                for r in related:
                    if r not in self.graph:
                        self._fetch_and_add_node(r, "related")
                        self.graph.add_edge(title, r, relation="related_to", confidence=0.7)
                    else:
                        if not self.graph.has_edge(title, r):
                            self.graph.add_edge(title, r, relation="related_to", confidence=0.7)
                    expanded.append(r)
                time.sleep(0.05)

        print(f"   初始圖譜：{self.graph.number_of_nodes()} 個節點，{self.graph.number_of_edges()} 條邊")

        # Step 4: [新增] 從文本抽取實體/關係並加入圖譜
        self._extract_and_add_relations()

        # Step 5: [新增] 社群偵測與摘要
        detector = CommunityDetector(self.graph)
        self.communities = detector.detect()
        self.community_summaries = detector.summarize_communities(self.graph)

        print(f"   最終圖譜：{self.graph.number_of_nodes()} 個節點，{self.graph.number_of_edges()} 條邊")
        
        # [NEW] Save to cache if enabled
        self._save_to_cache()

        # Step 6: 格式化
        return self._format_context(midi_features, unique_seeds)

    def _save_to_cache(self, cache_dir: str = "rag_cache"):
        """將圖譜儲存至本地，避免重複構建。"""
        import pickle
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "music_knowledge_graph.pkl")
        with open(cache_path, "wb") as f:
            pickle.dump(self.graph, f)

    def add_new_concepts(self, concepts: List[str]) -> Dict:
        """
        動態增加新概念並返回增量資料 (vis.js 格式)。
        """
        old_nodes = set(self.graph.nodes())
        old_edges = set(self.graph.edges())

        for concept in concepts:
            self._fetch_and_add_node(concept, "extracted_entity")
        
        # 執行關係抽取
        self._extract_and_add_relations()

        new_nodes = []
        for node, data in self.graph.nodes(data=True):
            if node not in old_nodes:
                new_nodes.append({
                    "id": node,
                    "label": node,
                    "group": data.get("type", "extracted_entity"),
                    "title": data.get("extract", "No description available.")
                })

        new_edges = []
        for u, v, data in self.graph.edges(data=True):
            if (u, v) not in old_edges:
                new_edges.append({
                    "from": u,
                    "to": v,
                    "label": data.get("relation", "related_to")
                })
        
        return {"nodes": new_nodes, "edges": new_edges}
        print(f"💾 圖譜已緩存至: {cache_path}")

    def load_from_cache(self, cache_dir: str = "rag_cache") -> bool:
        """載入已有的圖譜。"""
        import pickle
        cache_path = os.path.join(cache_dir, "music_knowledge_graph.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"📦 已從緩存載入圖譜 ({self.graph.number_of_nodes()} 節點)")
            return True
        return False

    def visualize(self, output_path: str = "graph_rag_viz.png", show_desc: bool = False):
        """將目前圖譜繪製並儲存為圖片。"""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if self.graph.number_of_nodes() == 0:
            print("⚠️ 圖譜為空，無法視覺化。")
            return

        # 加大畫布以容納描述文字
        fig, ax = plt.subplots(figsize=(20, 14), dpi=150)
        pos = nx.kamada_kawai_layout(self.graph) # 改用更開闊的佈局
        
        # 繪製邊與節點
        nx.draw_networkx_edges(self.graph, pos, alpha=0.45, edge_color='gray', ax=ax)
        
        node_colors = []
        labels = {}
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node].get('type', 'related')
            color = '#e74c3c' if ntype in ['seed', 'extracted_entity'] else '#3498db'
            node_colors.append(color)
            
            # 構建標籤文字 (增加換行)
            txt = node
            if show_desc:
                summary = self.graph.nodes[node].get('extract', '')
                if summary:
                    # 每 30 個字自動換行
                    desc = summary[:120].strip()
                    import textwrap
                    wrapped_desc = "\n".join(textwrap.wrap(desc, width=35))
                    txt = f"{node}\n({wrapped_desc}...)"
            labels[node] = txt

        nx.draw_networkx_nodes(self.graph, pos, node_size=1500, node_color=node_colors, alpha=0.9, ax=ax)
        
        # 繪製標籤 (加上白色背景邊框確保清晰)
        # 使用更穩定的 draw_networkx_labels 參數
        for node, txt in labels.items():
            x, y = pos[node]
            ax.text(x, y, txt, fontsize=9, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'),
                    zorder=10)

        plt.title("Professional Music Theory Knowledge Graph (GraphRAG)", fontsize=22, fontweight='bold', pad=30)
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close()
        print(f"🎨 視覺化圖譜（{'含描述' if show_desc else '僅名稱'}）已儲存至: {output_path}")




    # ── [新增] 從文本抽取實體/關係 ────────────────
    def _extract_and_add_relations(self):
        """
        遍歷圖中所有節點的文本摘要，抽取實體/關係。
        將新關係加入圖譜邊。
        """
        print("抽取實體/關係...")
        extracted_count = 0
        
        for node, data in list(self.graph.nodes(data=True)):
            extract_text = data.get("extract", "")
            if not extract_text:
                continue
            
            relations = self.extractor.extract_relations(extract_text)
            for e1, rel_type, e2 in relations:
                # 檢查實體是否已在圖中
                if e1 not in self.graph:
                    self.graph.add_node(e1, type="extracted_entity", extract="", source="relation")
                if e2 not in self.graph:
                    self.graph.add_node(e2, type="extracted_entity", extract="", source="relation")
                
                # 加入邊
                if not self.graph.has_edge(e1, e2):
                    self.graph.add_edge(e1, e2, relation=rel_type, confidence=0.5)
                    extracted_count += 1
        
        print(f"抽取完成：新增 {extracted_count} 條關係")

    def _fetch_and_add_node(self, concept: str, ntype: str):
        """查 Wikipedia，若失敗則改用 Gemini Google Search Grounding 作為後備"""
        if concept in self.graph:
            return

        if not self.use_web_search:
            self.graph.add_node(concept, type=ntype, extract="", source="local")
            return

        info = self.searcher.fetch_summary(concept)
        if info:
            self.graph.add_node(
                concept,
                type=ntype,
                extract=info.get("extract", ""),
                categories=info.get("categories", []),
                source="wikipedia",
                wiki_title=info.get("title", concept),
            )
        else:
            # Wikipedia 找不到 → 使用 Gemini Google Search grounding 作為後備
            print(f"   [RAG] Wikipedia miss for '{concept}'. Trying Gemini grounding...")
            try:
                from gemini_service import GeminiService
                gs = GeminiService()
                grounding_info = gs.search_with_grounding(concept)
                if grounding_info:
                    self.graph.add_node(
                        concept,
                        type=ntype,
                        extract=grounding_info.get("extract", ""),
                        categories=["Music Theory"],
                        source=grounding_info.get("source", "google_search"),
                        wiki_title=concept,
                    )
                    return
            except Exception as e:
                print(f"   [RAG] Gemini grounding also failed for '{concept}': {e}")
            # 最終後備：加空節點，避免邊懸空
            self.graph.add_node(concept, type=ntype, extract=f"No description found for '{concept}'.", source="not_found")

    # ── 格式化提示詞上下文 ────────────────────
    def _format_context(
        self,
        midi_features: List[str],
        seed_concepts: List[Tuple[str, str]],
    ) -> str:
        lines = ["## 知識圖谱上下文 (Graph RAG — Web Search + Community-Focused)\n"]
        lines.append(f"**偵測到的 MIDI 特徵**: {', '.join(midi_features)}\n")

        # 依類型分組節點
        type_groups: Dict[str, List[str]] = {}
        for node, data in self.graph.nodes(data=True):
            ntype = data.get("type", "other")
            type_groups.setdefault(ntype, []).append(node)

        TYPE_LABELS = {
            "technique":  "演奏技術 / 織體",
            "form":       "音樂形式",
            "instrument": "樂器",
            "theory":     "樂理概念",
            "related":    "延伸相關概念",
            "extracted_entity": "抽取實體",
            "other":      "其他",
            "not_found":  "(未找到 Wikipedia 頁面)",
        }

        for ntype, nodes in type_groups.items():
            if ntype == "not_found":
                continue
            label = TYPE_LABELS.get(ntype, ntype)
            lines.append(f"**{label}**: {', '.join(nodes)}")

        # 核心概念摘要（只取種子概念，避免過長）
        lines.append("\n**核心概念說明**:")
        for title, _ in seed_concepts[:6]:  # 最多 6 個
            if title in self.graph:
                node_data = self.graph.nodes[title]
                extract = node_data.get("extract", "")
                if extract:
                    # 只取第一段（到第一個換行或 300 字）
                    first_para = extract.split("\n")[0][:300]
                    lines.append(f"\n- **{title}**: {first_para}")

        # 圖谱邊（關係）
        edges = list(self.graph.edges(data=True))
        if edges:
            lines.append("\n**概念關聯**:")
            for src, tgt, edata in edges[:10]:  # 最多列出 10 條
                rel = edata.get("relation", "related_to").replace("_", " ")
                lines.append(f"  - {src} → {tgt}  [{rel}]")

        # [新增] Query-focused 社群聚合
        if self.communities and self.community_summaries:
            aggregator = QueryFocusedAggregator(
                self.graph,
                self.communities,
                self.community_summaries
            )
            aggregated = aggregator.aggregate_result(midi_features, top_k=3)
            lines.append("\n" + aggregated)

        lines.append("")
        return "\n".join(lines)


    # ── 工具方法 ──────────────────────────────
    def query(self, start_concepts: List[str], depth: int = 1) -> Dict:
        """
        通用查詢介面（與舊版相容）。
        直接觸發網路搜尋並回傳圖谱資訊。
        """
        self.get_analysis_context(start_concepts)

        all_nodes = {}
        for node, data in self.graph.nodes(data=True):
            all_nodes[node] = {
                "type": data.get("type"),
                "extract": data.get("extract", "")[:200],
                "source": data.get("source"),
            }

        relations = [
            {
                "from": s,
                "to": t,
                "relation": d.get("relation", "related"),
                "confidence": d.get("confidence", 1.0),
            }
            for s, t, d in self.graph.edges(data=True)
        ]

        return {
            "input_concepts": start_concepts,
            "concepts": all_nodes,
            "relations": relations,
            "total_nodes": self.graph.number_of_nodes(),
        }

    def to_json(self, output_file: str = "knowledge_graph.json"):
        """將目前圖谱匯出為 JSON"""
        data = {
            "nodes": [
                {
                    "name": n,
                    "type": d.get("type"),
                    "source": d.get("source"),
                    "extract_preview": d.get("extract", "")[:150],
                }
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"from": s, "to": t, "relation": d.get("relation", "related")}
                for s, t, d in self.graph.edges(data=True)
            ],
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"知識圖谱已匯出到 {output_file}")

    def visualize_subgraph(self, concepts: List[str], output_file: str = "graph.txt"):
        """ASCII 文字可視化（觸發網路搜尋後輸出）"""
        self.get_analysis_context(concepts)

        lines = ["=== 知識圖谱可視化 ===\n"]
        for node, data in self.graph.nodes(data=True):
            lines.append(f"【{node}】 (type={data.get('type')}, src={data.get('source')})")
            extract = data.get("extract", "")
            if extract:
                lines.append(f"  摘要: {extract[:120]}...")
            succs = list(self.graph.successors(node))
            if succs:
                lines.append(f"  → {', '.join(succs)}")
            lines.append("")

        output = "\n".join(lines)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"可視化已儲存到 {output_file}")
        return output

# 測試
if __name__ == "__main__":
    kg = MusicKnowledgeGraph()

    print("=== 測試：從 MIDI 特徵建圖 ===")
    ctx = kg.get_analysis_context(["Polyphony", "Keyboard", "RepetitiveForm"])
    print(ctx)

    print("\n=== 匯出 JSON ===")
    kg.to_json()