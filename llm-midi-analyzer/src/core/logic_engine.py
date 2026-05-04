from music21 import stream, chord, roman, voiceLeading, interval

class MusicLogicEngine:
    def __init__(self):
        pass
        
    def analyze_rna(self, midi_stream, key_str='C'):
        """自動化羅馬數字分析 (Roman Numeral Analysis)"""
        # Convert to chords
        chords = midi_stream.chordify()
        rna_results = []
        
        # Iterating through chords to map to Roman Numerals
        for c in chords.recurse().getElementsByClass('Chord'):
            try:
                rn = roman.romanNumeralFromChord(c, key_str)
                rna_results.append({
                    'offset': c.offset,
                    'figure': rn.figure,
                    'key': key_str
                })
            except Exception as e:
                # Handle unexpected chord structures gracefully
                pass
                
        return rna_results

    def check_parallel_fifths_reward(self, midi_stream):
        """
        利用 music21 檢查生成的分析報告是否與 MIDI 事實相符。
        尋找連續的平行五度，並計算一個 reward。
        """
        chords = midi_stream.chordify()
        penalty = 0.0
        prev_chord = None
        
        # 簡易平行五度與隱藏音程檢測邏輯
        for c in chords.recurse().getElementsByClass('Chord'):
            # Only consider chords with at least 2 pitches for interval evaluation
            if prev_chord and len(c.pitches) >= 2 and len(prev_chord.pitches) >= 2:
                # 這裡為了簡單，我們使用 music21 內建的 VoiceLeadingQuartet
                # 如果是四聲部，我們可以直接分析：
                if len(c.pitches) == 4 and len(prev_chord.pitches) == 4:
                    vlq = voiceLeading.VoiceLeadingQuartet(prev_chord, c)
                    if vlq.parallelFifth():
                        penalty -= 1.0 # 嚴重的平行五度懲罰
                    if vlq.hiddenFifth():
                        penalty -= 0.5 # 隱藏五度懲罰
            prev_chord = c
            
        # 若沒有平行五度，給予正向 reward
        return 1.0 if penalty == 0 else penalty

    def extract_hidden_intervals(self, midi_stream):
        """
        深入提取隱藏音程，支援 GRPO 獎勵金的精細化。
        """
        hidden_intervals = []
        chords = midi_stream.chordify()
        prev_chord = None
        for c in chords.recurse().getElementsByClass('Chord'):
             if prev_chord and len(c.pitches) == 4 and len(prev_chord.pitches) == 4:
                 vlq = voiceLeading.VoiceLeadingQuartet(prev_chord, c)
                 if vlq.hiddenFifth():
                     hidden_intervals.append({'offset': c.offset, 'type': 'hidden_fifth'})
                 if vlq.hiddenOctave():
                     hidden_intervals.append({'offset': c.offset, 'type': 'hidden_octave'})
             prev_chord = c
        return hidden_intervals
