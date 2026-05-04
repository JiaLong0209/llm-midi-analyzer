import os
import numpy as np
import matplotlib.pyplot as plt
import music21
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Dict

# ==========================================
# 0. 音程映射字典 (Interval Mapping)
# ==========================================
INTERVAL_MAP = {
    0: "P1", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4", 6: "TT", 7: "P5",
    8: "m6", 9: "M6", 10: "m7", 11: "M7", 12: "P8", 13: "m9", 14: "M9",
    15: "m10", 16: "M10", 17: "P11", 19: "P12", 21: "M13"
}

def get_interval_name(semitones: int) -> str:
    """將半音數量轉換為標準音程名稱"""
    abs_semi = abs(semitones)
    sign = "+" if semitones > 0 else "-" if semitones < 0 else ""
    
    if abs_semi in INTERVAL_MAP:
        name = INTERVAL_MAP[abs_semi]
    else:
        octaves = abs_semi // 12
        remainder = abs_semi % 12
        name = f"{octaves}8va+{INTERVAL_MAP.get(remainder, str(remainder))}"
        
    if abs_semi == 0:
        return "P1"
    return f"{sign}{name}"

# ==========================================
# 1. 攝取層：動機感知 (Motif Discovery via MTPs)
# ==========================================
class MotifDiscoverer:
    def __init__(self, score: music21.stream.Score):
        self.score = score
        self.time_slices = self._extract_time_slices()

    def _extract_time_slices(self) -> Dict[float, List[int]]:
        slices = {}
        for n in self.score.flatten().notes:
            offset = round(float(n.offset), 3)
            if offset not in slices:
                slices[offset] = []
                
            if n.isChord:
                for p in n.pitches:
                    slices[offset].append(p.midi)
            elif n.isNote:
                slices[offset].append(n.pitch.midi)
                
        for off in slices:
            slices[off] = sorted(list(set(slices[off])))
            
        return dict(sorted(slices.items()))

    def find_horizontal_motifs(self, window_size: int = 3) -> Dict[str, int]:
        offsets = list(self.time_slices.keys())
        if len(offsets) < window_size + 1: return {}
        skyline_pitches = [self.time_slices[off][-1] for off in offsets]
        patterns = {}
        for i in range(len(skyline_pitches) - window_size):
            pitches = skyline_pitches[i : i + window_size + 1]
            intervals = tuple(get_interval_name(pitches[j+1] - pitches[j]) for j in range(window_size))
            pattern_key = f"[{', '.join(intervals)}]"
            patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
        return {k: v for k, v in sorted(patterns.items(), key=lambda x: x[1], reverse=True) if v > 1}

    def find_vertical_motifs(self) -> Dict[str, int]:
        patterns = {}
        for off, pitches in self.time_slices.items():
            if len(pitches) > 1:
                bass = pitches[0]
                intervals = tuple(get_interval_name(p - bass) for p in pitches)
                pattern_key = f"[{', '.join(intervals)}]"
                patterns[pattern_key] = patterns.get(pattern_key, 0) + 1
        return {k: v for k, v in sorted(patterns.items(), key=lambda x: x[1], reverse=True) if v > 1}

# ==========================================
# 2. 多模態編碼層：鋼琴捲簾視覺化
# ==========================================
class PianoRollGenerator:
    def __init__(self, score: music21.stream.Score, ticks_per_beat: int = 4):
        self.score = score
        self.ticks_per_beat = ticks_per_beat 
        self.total_beats = int(np.ceil(self.score.highestTime))
        self.time_steps = max(1, self.total_beats * self.ticks_per_beat)
        self.matrix = np.zeros((88, self.time_steps), dtype=np.float32)

    def generate_matrix(self) -> np.ndarray:
        for n in self.score.flatten().notes:
            start_tick = int(n.offset * self.ticks_per_beat)
            end_tick = int((n.offset + n.quarterLength) * self.ticks_per_beat)
            start_tick = max(0, min(start_tick, self.time_steps - 1))
            end_tick = max(0, min(end_tick, self.time_steps))

            if n.isChord:
                for p in n.pitches:
                    self._fill_pitch(p.midi, start_tick, end_tick, n.volume.velocity)
            elif n.isNote:
                self._fill_pitch(n.pitch.midi, start_tick, end_tick, n.volume.velocity)
        return self.matrix

    def _fill_pitch(self, midi_pitch: int, start: int, end: int, velocity: int):
        if 21 <= midi_pitch <= 108:
            row_idx = 108 - midi_pitch
            vol = velocity if velocity is not None else 64
            self.matrix[row_idx, start:end] = vol / 127.0 

    def plot_image(self, save_path: str = "piano_roll.png"):
        plt.figure(figsize=(15, 5))
        plt.imshow(self.matrix, aspect='auto', cmap='magma', interpolation='nearest')
        plt.colorbar(label='Velocity (Normalized)')
        plt.title('2D Piano Roll Visual Representation (88 x Time)')
        plt.xlabel('Time Steps')
        plt.ylabel('MIDI Pitch (88 Keys)')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# ==========================================
# 3. 結構分析層：無監督自我相似性矩陣 (SSM)
# ==========================================
class SSMGenerator:
    """
    透過餘弦相似度計算時間窗口間的相似性，實作音樂曲式 (Musical Form) 的無監督分割。
    """
    def __init__(self, piano_roll: np.ndarray, window_size_steps: int = 16):
        self.piano_roll = piano_roll
        self.window_size = window_size_steps # 預設 16 個 step = 4 拍 (1 個 4/4 小節)
        self.ssm = self._compute_ssm()

    def _compute_ssm(self) -> np.ndarray:
        _, time_steps = self.piano_roll.shape
        num_windows = time_steps // self.window_size
        
        if num_windows < 2:
            return np.zeros((1, 1))

        features = []
        for i in range(num_windows):
            start = i * self.window_size
            end = start + self.window_size
            window = self.piano_roll[:, start:end]
            
            # 將該窗口內的時間軸壓扁，取得該小節的「音高分佈輪廓 (Pitch Profile)」
            pitch_profile = np.sum(window, axis=1)
            
            # 正規化特徵向量
            norm = np.linalg.norm(pitch_profile)
            if norm > 0:
                pitch_profile = pitch_profile / norm
            features.append(pitch_profile)
            
        features = np.array(features) # 維度: (小節數, 88鍵)
        
        # 使用 scipy 計算兩兩之間的餘弦距離 (Cosine Distance)
        # 相似度 = 1 - 距離
        distances = pdist(features, metric='cosine')
        ssm = 1 - squareform(distances)
        return ssm

    def plot_ssm(self, save_path: str = "ssm_matrix.png"):
        """繪製 SSM 熱力圖"""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.ssm, cmap='viridis', origin='lower', interpolation='nearest')
        plt.colorbar(label='Cosine Similarity')
        plt.title('Self-Similarity Matrix (SSM) for Musical Form Segmentation')
        plt.xlabel('Time Window (Measures)')
        plt.ylabel('Time Window (Measures)')
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"🖼️ [Structural Layer] SSM 矩陣影像已儲存至: {save_path}")
        plt.show()

# ==========================================
# 主程式測試執行
# ==========================================
def run_test(file_path: str):
    print(f"\n[*] 正在讀取 MIDI 檔案: {file_path}")
    if not os.path.exists(file_path): return
    score = music21.converter.parse(file_path)

    print("\n" + "="*60)
    print("1. 攝取與表徵層：多聲部動機發現 (Polyphonic Motif Discovery)")
    print("="*60)
    discoverer = MotifDiscoverer(score)
    h_motifs = discoverer.find_horizontal_motifs(window_size=6)
    print("\n【水平旋律動機】前 3 名:")
    for i, (pattern, count) in enumerate(list(h_motifs.items())[:3]):
        print(f"   - 動機 {i+1}: 走向 {pattern} -> 出現 {count} 次")
        
    v_motifs = discoverer.find_vertical_motifs()
    print("\n【垂直和聲動機】前 3 名:")
    for i, (pattern, count) in enumerate(list(v_motifs.items())[:3]):
        print(f"   - 結構 {i+1}: 音程 {pattern} -> 出現 {count} 次")

    print("\n" + "="*60)
    print("2. 視覺編碼層與結構分析：Piano Roll & SSM 曲式分割")
    print("="*60)
    
    # 產生鋼琴捲簾矩陣
    generator = PianoRollGenerator(score, ticks_per_beat=4)
    matrix = generator.generate_matrix()
    generator.plot_image("test_piano_roll.png")
    
    # 【核心創新】產生自我相似性矩陣 (SSM)
    # 假設 ticks_per_beat=4，一個 4/4 拍的小節就是 16 個 steps
    ssm_gen = SSMGenerator(matrix) 
    
    print("\n💡 [SSM 結構洞察]:")
    print("透過餘弦相似度矩陣，我們在本地算出了音樂的『區塊對稱性』。")
    print("若 SSM 熱力圖在左下到右上出現『明顯的方塊』，代表曲式為 A-B-A 結構。")
    print("這證明了本系統不需依賴 LLM，就能進行高等級的巨觀樂理推理！")
    
    ssm_gen.plot_ssm("test_ssm.png")

if __name__ == "__main__":
    midi_dir = "../midi/" if os.path.exists("../midi/") else "./midi/"
    if not os.path.exists(midi_dir):
        print(f"[!] 找不到 {midi_dir} 資料夾。")
    else:
        midis = [f for f in os.listdir(midi_dir) if f.lower().endswith(('.mid', '.midi'))]
        if not midis:
            print("[!] 資料夾內沒有 MIDI 檔案。")
        else:
            for i, midi in enumerate(midis): print(f"  [{i}] {midi}")
            try:
                index = int(input("\n請輸入編號 (Index): "))
                if 0 <= index < len(midis):
                    run_test(os.path.join(midi_dir, midis[index]))
            except ValueError:
                print("[!] 請輸入有效的數字。")