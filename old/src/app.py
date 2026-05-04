import os
import json
import threading
import http.server
import socketserver
import webbrowser
import shutil
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from typing import List, Dict, Any
from pathlib import Path

# Detect Project Root (Parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import music21
# Import the new Google GenAI SDK
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment from project root
load_dotenv(PROJECT_ROOT / ".env")

# ==========================================
# 0. SOLID Entities (Note Formatting)
# ==========================================
@dataclass
class NoteFormatConfig:
    show_velocity: bool = False
    show_offset: bool = False
    show_duration: bool = False
    show_chord_name: bool = True

class INoteFormatter(ABC):
    @abstractmethod
    def format_note(self, element: Any, config: NoteFormatConfig) -> List[str]:
        pass

class StandardNoteFormatter(INoteFormatter):
    """SOLID implementation of note formatting based on configuration"""
    def format_note(self, element: Any, config: NoteFormatConfig) -> List[str]:
        formatted = []
        
        # Helper to format a single pitch/note
        def _get_note_str(p: Any, el: Any) -> str:
            res = p.nameWithOctave
            
            # Conditionally add velocity
            if config.show_velocity:
                vol = el.volume.velocity if el.volume.velocity is not None else 64
                res = f"{vol}({res})"
            
            # Conditionally add duration
            if config.show_duration:
                try:
                    dur = round(float(el.quarterLength), 2)
                except:
                    dur = str(el.quarterLength)
                res = f"{res},{dur}"
            
            # Wrap in parens if we have extra info but no offset
            if (config.show_velocity or config.show_duration) and not config.show_offset:
                res = f"({res})"
            
            # Conditionally add offset (start time)
            if config.show_offset:
                try:
                    off = round(float(el.offset), 2)
                except:
                    off = str(el.offset)
                # Combine into final format: offset:info
                res = f"{off}:{res}"
            
            return res

        if element.isChord:
            for p in element.pitches:
                formatted.append(_get_note_str(p, element))
        elif element.isNote:
            formatted.append(_get_note_str(element.pitch, element))
            
        return formatted

# ==========================================
# 1. Configuration 
# ==========================================
@dataclass
class AppConfig:
    # llm_model_name: str = "models/gemma-3-27b-it" 
    llm_model_name: str = os.environ.get("LLM_MODEL_NAME", "models/gemma-3-27b-it")
    user_prompt: str = os.environ.get("USER_PROMPT", "請根據上述全局數據以及『每一軌』的細節音符，給予詳盡作曲手法、真實和弦判讀。務必嚴格遵守和弦記號規範！請不要只列出籠統的『M1-8』，必須明確指出『哪一個具體小節 (如 M14)』使用了什麼特定的作曲手法 (如模進 Sequence、持續音 Pedal)。所有和弦請附上級數 (如 Am (I), G (VII))。")
    api_key: str = os.environ.get("GOOGLE_API_KEY", "[ENCRYPTION_KEY]")
    llm_temperature: float = 0.3 
    midi_dir: str = str(PROJECT_ROOT / "midi")
    output_dir: str = str(PROJECT_ROOT / "output")
    default_midi_name: str = "default_song.mid"
    # Added options to reduce LLM token cost upon user request
    show_global_merged_chords: bool = False
    show_individual_instruments: bool = True
    
    # SOLID: Dependency Injection for note formatting
    note_format: NoteFormatConfig = field(default_factory=NoteFormatConfig)

    # Don't remove this system prompt
    # system_prompt: str = (
    #     "你是一位世界頂尖的古典與現代音樂作曲家、樂理大師。\n"
    #     "請根據以下萃取出的 MIDI 數據（包含全局特徵與按小節拆分的音軌數據），"
    #     "分析這首曲子的：調式 (Key)、音階 (Scale)、和弦進行 (Chord Progression)、"
    #     "音樂風格 (Music Style)、使用的作曲手法 (Composition Techniques)，以及樂器編制。\n\n"
    #     "【⚠️ 分析步驟與思維鏈 (Chain-of-Thought) 嚴格要求 ⚠️】\n"
    #     "在開始撰寫正式報告前，你必須在內部進行以下邏輯推演（請將推演過程簡要寫在報告開頭的「樂理基礎檢核」段落）：\n"
    #     "1. 確認當前全局調性 (Key)。\n"
    #     "2. 列出該調性的所有「順階和弦 (Diatonic Chords)」。(例如：若為 A minor，必須列出 Am, Bdim, C, Dm, Em, F, G)。\n"
    #     "3. 當分析到具體小節的 `[Chord: X]` 時，必須先將其與你的順階和弦清單核對。\n\n"
    #     "【⛔ 嚴格防呆與糾錯指南 (Negative Prompts) ⛔】\n"
    #     "1. 絕對禁止盲目套用爵士公式！如果你看到小七和弦接大七或屬七 (如 Gm7 -> Cmaj7)，在稱其為 ii-V-I 之前，必須先核對 Gm7 是否為當前調性的第二級！如果當前是 A 小調，Gm7 絕對不是 ii 級，請分析它是否為借用和弦 (Borrowed Chord)、次屬和弦的延伸，或是調性暫時轉移。\n"
    #     "2. 系統提供的 `[Chord: X]` 是經過演算法初步計算的主導和弦，請以此為基礎，配合後方的音符 (Notes) 來確認和聲張力，但不要在報告中像流水帳一樣列出每個 MIDI note。\n\n"
    #     "【📝 格式與術語規範】\n"
    #     "1. 所有和弦名稱【絕對只能使用標準英文記號】（例如：Am, Cmaj7, G7, F#m7, Bdim）。\n"
    #     "2. 絕對禁止將和弦翻譯成中文（❌ 嚴禁輸出「A小調和弦」、「C大三和弦」等字眼）。\n"
    #     "3. 在提到音階或調式時，請優先使用英文（如 A minor scale, C Dorian）。\n"
    #     "4. 若樂曲中途有 `[Key_Change: ...]` 或 `[TimeSig_Change: ...]` 標籤，請務必在報告中指出該小節的轉調或變拍手法。\n"
    #     "5. 報告需包含：旋律走向特色、多聲部互動、和弦進行的張力與解決、潛在的作曲手法（如模進、經過音、持續音 Pedal Point 等）。\n"
    #     "請用專業、嚴謹但生動易懂的文字輸出最終分析報告。"
    # )

    # system_prompt: str = (
    #     "你是一位具備通才能力的「全知全能音樂分析大師」，精通從中世紀對位法到現代實驗爵士的所有音樂風格。\n"
    #     "請根據提供的「時間序列切片 (Time-Slice)」數據，對這首 MIDI 進行深度解構。\n\n"
    #     "【🔍 汎用型分析邏輯 (Generalization Logic)】\n"
    #     "1. 垂直分析 (Vertical Harmony)：觀察每個時間點 (@offset) 的音程組合 (Intervals)。\n"
    #     "   - 若音程和諧，請分析其和弦功能 (Function)。\n"
    #     "   - 若音程不和諧，請分析其張力 (Tension) 是如何產生的，以及如何解決。\n"
    #     "2. 水平分析 (Horizontal Voice Leading)：觀察各聲部的獨立性與流動。是否存在主題模仿 (Imitation)、模進 (Sequence) 或對位關係？\n"
    #     "3. 織體分析 (Texture)：判斷這首曲子是單聲部 (Monophony)、主調和聲 (Homophony) 還是複調對位 (Polyphony)。\n\n"
    #     "【🚫 嚴格行為準則】\n"
    #     "- 嚴格遵守「樂理基礎檢核」：先列出全局調性的順階音階，再開始對比。不要被複雜的和弦名稱誤導。\n"
    #     "- 禁止過度解釋：如果數據顯示只有兩個音，就分析其「音程關係」，不要硬湊一個複雜的和弦名稱。\n"
    #     "- 術語規範：所有音樂術語、和弦、音程名稱【必須使用標準英文】(如 Major 3rd, Dm7, Diminished, Syncopation)。\n\n"
    #     "【📝 報告結構】\n"
    #     "一、風格鑑定：根據音符密度、動態、和聲語言判斷其最接近的音樂流派。\n"
    #     "二、結構與織體：描述曲式的發展與多聲部之間的互動關係。\n"
    #     "三、核心分析：結合垂直音程與水平旋律，分析其作曲手法（如對位、變奏、節奏特色等）。\n"
    #     "四、藝術評價：這首作品在該風格下表現出的創意與藝術價值。"
    # )
    system_prompt: str = (
        "你是一位精通「結構主義」的音樂分析大師，擅長從極簡數據中洞察複雜的對位邏輯。\n"
        "請根據提供的「垂直切片格點 (Counterpoint Grid)」數據進行分析。\n\n"
        "【🎹 數據閱讀指南】\n"
        "數據格式為 `時間:音符+音符(音程)`。例如 `1.0:C4+B4(M7)` 代表在 1.0 拍處，聲部形成了「大七度」的張力。\n\n"
        "【🔍 分析重點】\n"
        "1. 對位邏輯：觀察聲部間的音程變化。是從不協和（如 m2, M7）解決到協和（如 P5, M3）嗎？\n"
        "2. 旋律骨架與具體小節：分析旋律的「動機 (Motif)」與「作曲手法」。**絕對不能只含糊地寫 M1-M8，必須精確指出哪一個小節發生了什麼事** (例如：M5 出現了模進，M12 有持續音)。\n"
        "3. 和弦級數 (Roman Numerals)：所有提到的和弦，**必須**附帶其在當前調性下的級數分析 (例如：Am (i), Fmaj7 (VI), G7 (V7))。\n"
        "4. 織體密度：觀察每個小節的切片數量，判斷節奏的複雜度與風格。\n\n"
        "【⚠️ 嚴格規範】\n"
        "- 保持專業英文術語 (Am, Major 3rd, Counterpoint, Sequence)。\n"
        "- 報告需包含：風格鑑定、結構分析、核心對位手法 (精確標明小節)、以及作品的藝術評價。\n"
        "請用簡潔有力、充滿洞見的文字輸出報告。"
        "\n\n【📝 報告結構】\n"
        "一、風格鑑定：根據音符密度、動態、和聲語言判斷其最接近的音樂流派。\n"
        "二、具體結構與織體 (精確到小節)：列出明確的段落劃分與關鍵小節的互動關係。\n"
        "三、核心分析 (附帶級數)：結合垂直音程與水平旋律，分析其具體的作曲手法（精確到小節號），並使用級數標示和弦。\n"
        "四、藝術評價：這首作品在該風格下表現出的創意與藝術價值。"
        "請使用繁體中文輸出報告，但是和弦名稱、音階名稱、調式名稱、作曲手法名稱等專業術語請使用英文。"
    )

# ==========================================
# 2. Interfaces 
# ==========================================
class IMessageHistory(ABC):
    @abstractmethod
    def add_message(self, role: str, content: str):
        pass

    @abstractmethod
    def get_messages(self) -> List[Any]:
        pass

class IMidiAnalyzer(ABC):
    @abstractmethod
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        pass

class ILLMService(ABC):
    @abstractmethod
    def generate_analysis(self, history: IMessageHistory) -> str:
        pass

class IMidiPlayer(ABC):
    @abstractmethod
    def play(self, file_path: str):
        pass
    
    @abstractmethod
    def stop(self):
        pass

# ==========================================
# 3. Implementations 
# ==========================================
class InMemoryMessageHistory(IMessageHistory):
    def __init__(self, max_pairs: int = 5):
        self.messages = []
        self.max_pairs = max_pairs

    def add_message(self, role: str, content: str):
        self.messages.append({'role': role, 'content': content})
        self._trim()

    def get_messages(self) -> List[Any]:
        return self.messages

    def _trim(self):
        # Always keep the first pair (Context + First LLM Analysis)
        # Trim older messages if length exceeds (1 initial pair + max_pairs) * 2
        limit = 2 + (self.max_pairs * 2)
        while len(self.messages) > limit:
            # Pop the oldest interactive turn (index 2)
            self.messages.pop(2)

class WebVisualMidiPlayer(IMidiPlayer):
    """Visual Piano Roll MIDI Player using html-midi-player"""
    def __init__(self, port: int = 0):
        self.port = port
        self.server = None
        self.server_thread = None
        self.html_file = str(PROJECT_ROOT / "midi_player.html")

    def _generate_html(self, midi_filename: str):
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visual Piano Roll MIDI Player</title>
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem;
            margin: 0;
        }}
        .container {{
            width: 95%;
            max-width: 1400px;
        }}
        h2 {{
            text-align: center;
            color: #4CAF50;
        }}
        h3 {{
            color: #66BB6A;
            margin-top: 1.5rem;
            margin-bottom: 0.8rem;
        }}
        midi-player {{
            display: block;
            width: 100%;
            margin-bottom: 2rem;
        }}
        midi-visualizer {{
            display: block;
            width: 100%;
            height: 600px;
            background: #1e1e1e;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Visual Piano Roll MIDI Player</h2>
        <midi-player
            src="{midi_filename}"
            sound-font
            visualizer="#pianoRollVisualizer, #staffVisualizer">
        </midi-player>


        <midi-visualizer type="piano-roll" id="pianoRollVisualizer"></midi-visualizer>
        <midi-visualizer type="staff" id="staffVisualizer"></midi-visualizer>


</body>
</html>"""
        with open(self.html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _create_server(self):
        root_dir = str(PROJECT_ROOT)
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root_dir, **kwargs)
            def log_message(self, format, *args):
                pass # Suppress logs to keep console clean

        socketserver.TCPServer.allow_reuse_address = True
        self.server = socketserver.TCPServer(("127.0.0.1", self.port), Handler)
        self.port = self.server.server_address[1] # Update to dynamically assigned port

    def play(self, file_path: str):
        try:
            # Use relative path from PROJECT_ROOT so the local server can find it
            rel_midi_path = os.path.relpath(file_path, PROJECT_ROOT)
            
            self._generate_html(rel_midi_path)

            if not self.server:
                self._create_server()
                self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
                self.server_thread.start()

            url = f"http://127.0.0.1:{self.port}/{self.html_file}"
            print(f"▶️ [Player launched] Web server started! Dual Visualizers (Piano Roll & Staff) ready.")
            print(f"   👉 URL: {url}")
            # webbrowser.open(url) # Disabled as per requirement
        except Exception as e:
            print(f"[!] Playback failed: {e}")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        # Cleanup temporary HTML file
        if os.path.exists(self.html_file):
            try:
                os.remove(self.html_file)
            except:
                pass
                
        print("⏹️ [Player stopped] Web server closed.")


class Music21MidiAnalyzer(IMidiAnalyzer):
    def __init__(self, config: AppConfig, note_formatter: INoteFormatter = StandardNoteFormatter()):
        self.config = config
        self.note_formatter = note_formatter

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        print(f"[*] Deep scanning MIDI file (Full track extraction): {file_path} ...")
        t0 = time.time()
        try:
            score = music21.converter.parse(file_path)
            
            # --- Macro Features ---
            key_sig = score.analyze('key')
            
            time_signatures = score.flat.getElementsByClass(music21.meter.TimeSignature)
            ts = time_signatures[0].ratioString if time_signatures else "4/4 (Default)"
            
            tempos = score.flat.getElementsByClass(music21.tempo.MetronomeMark)
            bpm = tempos[0].number if tempos else "Unknown"

            parts = score.parts
            total_measures = len(parts[0].getElementsByClass('Measure')) if parts else 0

            # --- Restored Macro Features ---
            instrument_parts = music21.instrument.partitionByInstrument(score)
            instruments = [str(p.partName) for p in instrument_parts.parts if p.partName] if instrument_parts else ["Unknown"]

            chordified = score.chordify()
            chord_progression_by_8m = []
            current_8m_chords = []
            
            for m in chordified.getElementsByClass('Measure'):
                m_num = m.number
                chords_in_m = m.getElementsByClass('music21.chord.Chord')
                if chords_in_m:
                    c = chords_in_m[0]
                    try:
                        rn = music21.roman.romanNumeralFromChord(c, key_sig)
                        try:
                            cn = music21.harmony.chordSymbolFigureFromChord(c)
                            if cn == 'Chord Symbol Cannot Be Identified':
                                cn = c.pitchedCommonName
                        except:
                            cn = c.pitchedCommonName
                        chord_name = f"{rn.figure} ({cn})"
                    except:
                        chord_name = c.pitchedCommonName
                else:
                    chord_name = "NC"
                
                # Deduplicate consecutive identical chords within the current 8 measures block
                if not current_8m_chords or current_8m_chords[-1] != chord_name:
                    current_8m_chords.append(chord_name)
                
                if m_num % 8 == 0 or m_num == total_measures:
                    # Save block
                    start_m = max(1, m_num - (m_num - 1) % 8)
                    prog_str = " -> ".join(current_8m_chords)
                    chord_progression_by_8m.append(f"M{start_m}-{m_num}: {prog_str}")
                    current_8m_chords = []

            pitches = [p.nameWithOctave for p in score.flat.pitches]
            if pitches:
                low_note = min(score.flat.pitches)
                high_note = max(score.flat.pitches)
                pitch_range = f"{low_note.nameWithOctave} ~ {high_note.nameWithOctave}"
            else:
                pitch_range = "無音符數據"

            total_notes = len(score.flat.notes)
            duration = score.flat.highestTime
            density = total_notes / duration if duration > 0 else 0

            # --- Micro Features (Full Track Notes/Chords per Measure) ---
            tracks_data = {}
            
            # 1. Global Merged Chords (Time-Slice Analysis)
            tracks_data["Global_Merged_Chords"] = {}
            current_time_sig = None
            current_key_sig = None
            
            chordified_measures = list(chordified.getElementsByClass('Measure'))
            for measure in tqdm(chordified_measures, desc="Analyzing Global Chords"):
                m_number = measure.number
                notes_in_m = []
                
                m_time_sigs = measure.getElementsByClass(music21.meter.TimeSignature)
                if m_time_sigs:
                    new_ts = m_time_sigs[0].ratioString
                    if current_time_sig is None:
                        if ts and new_ts != str(ts):
                            notes_in_m.append(f"[TimeSig_Change: {new_ts}]")
                        current_time_sig = new_ts
                    elif new_ts != current_time_sig:
                        current_time_sig = new_ts
                        notes_in_m.append(f"[TimeSig_Change: {current_time_sig}]")

                m_key_sigs = measure.getElementsByClass(music21.key.KeySignature)
                if m_key_sigs:
                    if hasattr(m_key_sigs[0], 'asKey'):
                        new_ks = str(m_key_sigs[0].asKey())
                    else:
                        new_ks = str(m_key_sigs[0])
                        
                    if current_key_sig is None:
                        current_key_sig = new_ks
                    elif new_ks != current_key_sig:
                        current_key_sig = new_ks
                        notes_in_m.append(f"[Key_Change: {current_key_sig}]")

                slice_strings = []
                prev_top_pitch = None
                motion_count = {"Up": 0, "Down": 0, "Static": 0}
                
                offsets_dict = {}
                for element in measure.flatten().getElementsByClass('Chord'):
                    offset = round(float(element.offset), 3)
                    # Remove trailing zeros if it's cleanly rounded
                    offset = int(offset) if offset == int(offset) else offset
                    
                    if offset not in offsets_dict:
                        offsets_dict[offset] = element
                        
                for offset in sorted(offsets_dict.keys()):
                    c = offsets_dict[offset]
                    if not c.pitches:
                        continue
                        
                    sorted_pitches = sorted(c.pitches)
                    bass_pitch = sorted_pitches[0]
                    
                    chord_prefix = ""
                    if self.config.note_format.show_chord_name:
                        try:
                            chord_sym = music21.harmony.chordSymbolFigureFromChord(c)
                            if chord_sym and chord_sym != 'Chord Symbol Cannot Be Identified' and 'pedal' not in chord_sym.lower():
                                try:
                                    rn = music21.roman.romanNumeralFromChord(c, key_sig).figure
                                    chord_prefix = f"[{rn}({chord_sym})] "
                                except:
                                    chord_prefix = f"[{chord_sym}] "
                        except:
                            pass
                            
                    notes_str = "+".join([p.nameWithOctave for p in sorted_pitches])
                    
                    interval_names = []
                    if len(sorted_pitches) > 1:
                        for p in sorted_pitches[1:]:
                            try:
                                interval = music21.interval.Interval(bass_pitch, p).name
                                interval_names.append(interval)
                            except:
                                pass
                                
                    interval_str = f"({','.join(interval_names)})" if interval_names else ""
                    slice_strings.append(f"{chord_prefix}{offset}:{notes_str}{interval_str}")
                    
                    top_pitch = sorted_pitches[-1].midi
                    if prev_top_pitch is not None:
                        if top_pitch > prev_top_pitch: motion_count["Up"] += 1
                        elif top_pitch < prev_top_pitch: motion_count["Down"] += 1
                        else: motion_count["Static"] += 1
                    prev_top_pitch = top_pitch

                if not slice_strings:
                    slice_strings = ["Rest"]
                    
                meta_tags = []
                for n in notes_in_m:
                    if n.startswith("[Key_Change:") or n.startswith("[TimeSig_Change:"):
                        meta_tags.append(n)
                
                meta_str = " ".join(meta_tags) + " " if meta_tags else ""
                slices_combined = " | ".join(slice_strings)
                tracks_data["Global_Merged_Chords"][f"M{m_number}"] = f"{meta_str}{slices_combined}"
                
            # 2. Individual Instruments
            for i, part in enumerate(tqdm(parts, desc="Analyzing Instruments")):
                base_part_name = part.partName or f"Unknown_Instrument_{i+1}"
                
                # Ensure unique track names to prevent overwriting
                part_name = base_part_name
                suffix = 1
                while part_name in tracks_data:
                    part_name = f"{base_part_name}_{suffix}"
                    suffix += 1
                    
                tracks_data[part_name] = {}
                
                measures = list(part.getElementsByClass('Measure'))
                for measure in tqdm(measures, desc=f"Track: {part_name[:15]}", leave=False):
                    m_number = measure.number
                    notes_in_m = []
                    
                    for element in measure.flatten().notes:
                        offset = round(float(element.offset), 3)
                        offset = int(offset) if offset == int(offset) else offset
                        
                        if element.isChord:
                            notes_str = "+".join([p.nameWithOctave for p in sorted(element.pitches)])
                            notes_in_m.append(f"{offset}:{notes_str}")
                        elif element.isNote:
                            notes_in_m.append(f"{offset}:{element.nameWithOctave}")
                            
                    if not notes_in_m:
                        notes_in_m = ["Rest"]
                        
                    tracks_data[part_name][f"M{m_number}"] = " | ".join(notes_in_m)

            t1 = time.time()
            
            # Filter output based on config to save tokens
            final_tracks_data = {}
            if self.config.show_global_merged_chords and "Global_Merged_Chords" in tracks_data:
                final_tracks_data["Global_Merged_Chords"] = tracks_data["Global_Merged_Chords"]
            
            if self.config.show_individual_instruments:
                for k, v in tracks_data.items():
                    if k != "Global_Merged_Chords":
                        final_tracks_data[k] = v
            
            return {
                "estimated_key": str(key_sig),
                "mode": key_sig.mode,
                "time_signature": ts,
                "bpm": bpm,
                "total_measures": total_measures,
                "instruments": instruments,
                "pitch_range": pitch_range,
                "note_count": total_notes,
                "density": round(density, 2),
                "chord_progression_by_8m": chord_progression_by_8m,
                "detailed_tracks": final_tracks_data,
                "analysis_time_sec": round(t1 - t0, 3)
            }

        except Exception as e:
            print(f"[!] MIDI Parsing Failed: {e}")
            return {"error": str(e)}

    def print_visualization(self, data: Dict[str, Any]):
        print("\n" + "="*60)
        print("🎵 MIDI Deep Analysis Completed 🎵")
        print("="*60)
        print(f"🎹 Key: {data.get('estimated_key')} ({data.get('mode')} mode)")
        print(f"⏱️  Time: {data.get('time_signature')} | BPM: {data.get('bpm')}")
        print(f"📏 Total Measures: {data.get('total_measures')} | Note Count: {data.get('note_count')} | Density: {data.get('density')} notes/beat")
        print(f"🎚️  Pitch Range: {data.get('pitch_range', 'Unknown')} | 🎻 Instruments: {', '.join(data.get('instruments', []))}")

        print("\n🎼 Chords (Per 8 Measures):")
        for prog in data.get('chord_progression_by_8m', []):
            print(f"   {prog}")
        print("\n🎼 Full-track data preview (truncated for display):")
        for instr, measures in data.get('detailed_tracks', {}).items():
            # Create a simple preview string connecting first few measures
            preview_str = " | ".join([f"{m_num}: {notes}" for m_num, notes in list(measures.items())[:5]])
            preview = preview_str + " ... (truncated)" if len(measures) > 5 else preview_str
            print(f"\n🎷 Instrument: {instr}\n   => {preview}")
        print(f"\n⏱️  MIDI Parsing Time: {data.get('analysis_time_sec', 0)} seconds")
        print("="*60 + "\n")


class GoogleGenAIService(ILLMService):
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = genai.Client(api_key=self.config.api_key)

    def generate_analysis(self, history: IMessageHistory) -> str:
        print(f"[*] Prompting LLM ({self.config.llm_model_name}) ...")
        
        contents = []
        for msg in history.get_messages():
            contents.append(
                types.Content(
                    role=msg['role'], 
                    parts=[types.Part.from_text(text=msg['content'])]
                )
            )
        
        try:
            response = self.client.models.generate_content(
                model=self.config.llm_model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.config.llm_temperature
                )
            )
            # Print token usage
            if response.usage_metadata:
                usage = response.usage_metadata
                print(f"   [Tokens] Input: {usage.prompt_token_count} | Output: {usage.candidates_token_count} | Total: {usage.total_token_count}")
            
            return response.text
        except Exception as e:
            return f"[!] LLM failed: {e}"

# ==========================================
# 4. Orchestrator 
# ==========================================
class MusicAnalysisApp:
    def __init__(self, config: AppConfig, midi_analyzer: IMidiAnalyzer, llm_service: ILLMService, player: IMidiPlayer, history: IMessageHistory):
        self.config = config
        self.midi_analyzer = midi_analyzer
        self.llm_service = llm_service
        self.player = player
        self.history = history

    def select_midi_file(self) -> str:
        # Create midi dir if not exists
        if not os.path.exists(self.config.midi_dir):
            os.makedirs(self.config.midi_dir)

        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.askopenfilename(
            initialdir=os.path.abspath(self.config.midi_dir),
            title="請選擇一首 MIDI 檔案",
            filetypes=[("MIDI files", "*.mid *.midi *.MID", ), ("All files", "*.*")]
        )
        if not file_path:
            default_path = os.path.join(self.config.midi_dir, self.config.default_midi_name)
            print(f"[*] No file selected, falling back to: {default_path}")
            return default_path
        return file_path

    def run(self):
        file_path = self.select_midi_file()
        if not os.path.exists(file_path):
            print(f"[!] File not found: {file_path}")
            return

        # 1. Analyze MIDI 
        midi_data = self.midi_analyzer.analyze_file(file_path)
        if "error" in midi_data:
            return
            
        # 2. Save JSON Data
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(self.config.output_dir, base_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        json_path = os.path.join(output_dir, f"{base_name}_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(midi_data, f, ensure_ascii=False, indent=4)
        print(f"💾 [Data Saved] Analyzed MIDI data saved to: {json_path}")

        # 3. Console preview
        if isinstance(self.midi_analyzer, Music21MidiAnalyzer):
            self.midi_analyzer.print_visualization(midi_data)

        # 4. Play visual web piano roll
        self.player.play(file_path)

        # 5. Pause for user review and optional prompt additions
        print("\n[AI 分析準備就緒]")
        print("您可以在此時前往 output/ 資料夾查看生成的 JSON 分析檔案。")
        try:
            extra_prompt = input("若想讓 LLM 分析特定方向，請輸入補充提示詞 (直接按 Enter 將繼續分析，按 Ctrl+C 結束程式): ")
        except KeyboardInterrupt:
            print("\n[!] 已取消 AI 分析。")
            self.player.stop()
            return
            
        user_prompt = self.config.user_prompt
        if extra_prompt.strip():
            user_prompt += f"\n\n【使用者補充需求】: {extra_prompt.strip()}"

        # 6. Initialize LLM context
        compact_context = json.dumps(midi_data, ensure_ascii=False, separators=(',', ':'))
        initial_prompt = (
            f"【系統指令】\n{self.config.system_prompt}\n\n"
            f"【客觀樂理數據 (Compact Data)】\n{compact_context}\n\n"
            f"【使用者首發指令】\n{user_prompt}"
        )
        self.history.add_message("user", initial_prompt)
        
        print("\n🧠 [AI 音樂大師分析報告] 🧠")
        print("正在分析中...\n" + "-" * 60)
        
        t0_llm = time.time()
        analysis_report = self.llm_service.generate_analysis(self.history)
        t1_llm = time.time()
        llm_time = round(t1_llm - t0_llm, 3)
        
        self.history.add_message("model", analysis_report)
        
        print(analysis_report)
        print("-" * 60)
        print(f"⏱️  LLM 回覆時間 (LLM Response Time): {llm_time} 秒\n")

        # 7. Save AI Analysis Report
        report_path = os.path.join(output_dir, f"{base_name}_analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# MIDI Music Analysis Report: {base_name}\n\n")
            f.write(analysis_report)
        print(f"📄 [Report Saved] AI analysis report saved to: {report_path}")

        # 8. Interactive Chat Loop
        print("\n💬 [AI 音樂分析聊天室開啟]")
        print("您可以繼續對這首曲子進行提問，AI 會記住上方的分析與數據。")
        print("輸入 'exit' 或按下 Ctrl+C 結束。")
        while True:
            try:
                user_msg = input("\n[user]: ")
                if user_msg.strip().lower() in ['exit', 'quit']:
                    print("👋 結束聊天。")
                    break
                if not user_msg.strip():
                    continue
                    
                self.history.add_message("user", user_msg)
                print("[AI]: 正在思考...")
                
                t0_chat = time.time()
                reply = self.llm_service.generate_analysis(self.history)
                t1_chat = time.time()
                chat_time = round(t1_chat - t0_chat, 3)
                
                self.history.add_message("model", reply)
                print(f"\n🧠 {reply}")
                print(f"\n⏱️  LLM 回覆時間: {chat_time} 秒")
                
            except KeyboardInterrupt:
                print("\n[!] 取消對話，結束聊天。")
                break

        # 9. Wait to cleanly shutdown server
        self.player.stop()

# ==========================================
# 5. Main Execution
# ==========================================
def main():
    config = AppConfig()
    # SOLID: Inject formatter and config
    formatter = StandardNoteFormatter()
    midi_analyzer = Music21MidiAnalyzer(config, formatter)
    llm_service = GoogleGenAIService(config)
    player = WebVisualMidiPlayer()
    history = InMemoryMessageHistory(max_pairs=5)
    
    app = MusicAnalysisApp(config, midi_analyzer, llm_service, player, history)
    app.run()

if __name__ == "__main__":
    main()