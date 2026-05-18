import os
import time
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import music21

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


class Music21MidiAnalyzer:
    def __init__(self, show_global_merged_chords: bool = False, show_individual_instruments: bool = True):
        self.show_global_merged_chords = show_global_merged_chords
        self.show_individual_instruments = show_individual_instruments
        self.note_formatter = StandardNoteFormatter()
        self.note_format_config = NoteFormatConfig()

    def analyze_file(self, file_path: str, start_measure: int = None, end_measure: int = None) -> Dict[str, Any]:
        print(f"[*] Deep scanning MIDI file with music21: {file_path} ...")
        t0 = time.time()
        try:
            score = music21.converter.parse(file_path)
            
            # --- Macro Features ---
            key_sig = score.analyze('key')
            
            time_signatures = score.flat.getElementsByClass(music21.meter.TimeSignature)
            ts = time_signatures[0].ratioString if time_signatures else "4/4 (Default)"
            
            tempo_map = []
            try:
                import pretty_midi
                pm = pretty_midi.PrettyMIDI(file_path)
                times, bpms = pm.get_tempo_changes()
                
                if len(times) > 0:
                    current_beat = 0.0
                    current_time = times[0]
                    current_bpm = bpms[0]
                    
                    tempo_map.append({
                        "beat": 0.0,
                        "time": 0.0,
                        "bpm": round(float(current_bpm), 4)
                    })
                    
                    for next_time, next_bpm in zip(times[1:], bpms[1:]):
                        delta_time = next_time - current_time
                        if delta_time > 0.0001:
                            delta_beat = delta_time * (current_bpm / 60.0)
                            current_beat += delta_beat
                            current_time = next_time
                            current_bpm = next_bpm
                            tempo_map.append({
                                "beat": round(float(current_beat), 4),
                                "time": round(float(current_time), 4),
                                "bpm": round(float(current_bpm), 4)
                            })
                else:
                    tempo_map.append({
                        "beat": 0.0,
                        "time": 0.0,
                        "bpm": 120.0
                    })
            except Exception as e:
                print(f"[!] Warning: failed to parse tempo map: {e}")
                tempo_map = [{"beat": 0.0, "time": 0.0, "bpm": 120.0}]
            
            bpm = tempo_map[0]["bpm"] if tempo_map else 120.0
            
            tempo_changes_desc = []
            if len(tempo_map) > 1:
                measure_changes = {}
                ts_numerator = time_signatures[0].numerator if time_signatures and time_signatures[0].numerator else 4
                for entry in tempo_map:
                    beat = entry["beat"]
                    bpm_val = round(entry["bpm"], 2)
                    
                    # Find measure number corresponding to the beat offset
                    m_num = 1
                    for m in score.recurse().getElementsByClass(music21.stream.Measure):
                        if m.offset <= beat < m.offset + m.duration.quarterLength:
                            m_num = m.number
                            break
                    else:
                        m_num = int(beat / ts_numerator) + 1
                    
                    if m_num not in measure_changes:
                        measure_changes[m_num] = []
                    measure_changes[m_num].append((beat, bpm_val))
                
                # Build aggregated descriptions
                for m_num, changes in sorted(measure_changes.items()):
                    if len(changes) > 5:
                        bpms = [c[1] for c in changes]
                        min_bpm = min(bpms)
                        max_bpm = max(bpms)
                        start_bpm = changes[0][1]
                        end_bpm = changes[-1][1]
                        tempo_changes_desc.append(
                            f"Measure {m_num}: Expressive tempo variations / Rubato (BPM ranges from {min_bpm} to {max_bpm}; starting at {start_bpm}, ending at {end_bpm})"
                        )
                    else:
                        for beat, bpm_val in changes:
                            tempo_changes_desc.append(f"Measure {m_num} (Beat {round(beat, 2)}): BPM changes to {bpm_val}")
            else:
                tempo_changes_desc.append(f"Static BPM of {bpm} throughout the piece.")

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
                if start_measure is not None and m_num < start_measure:
                    continue
                if end_measure is not None and m_num > end_measure:
                    continue

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
                
                is_last = (m_num == total_measures) or (end_measure is not None and m_num == end_measure)
                if m_num % 8 == 0 or is_last:
                    # Save block
                    start_m = max(start_measure or 1, m_num - (m_num - 1) % 8)
                    prog_str = " -> ".join(current_8m_chords)
                    chord_progression_by_8m.append(f"M{start_m}-{m_num}: {prog_str}")
                    current_8m_chords = []

            # Bound-aware note and pitch calculations
            sliced_notes = []
            for note in score.flat.notes:
                try:
                    m_num = note.measureNumber
                    if m_num is None:
                        continue
                except:
                    continue
                if start_measure is not None and m_num < start_measure:
                    continue
                if end_measure is not None and m_num > end_measure:
                    continue
                sliced_notes.append(note)
            
            pitches = []
            for element in sliced_notes:
                if element.isChord:
                    pitches.extend(element.pitches)
                elif element.isNote:
                    pitches.append(element.pitch)

            if pitches:
                low_note = min(pitches)
                high_note = max(pitches)
                pitch_range = f"{low_note.nameWithOctave} ~ {high_note.nameWithOctave}"
            else:
                pitch_range = "無音符數據"

            total_notes = len(sliced_notes)
            duration = score.flat.highestTime
            density = total_notes / duration if duration > 0 else 0

            # --- Micro Features (Full Track Notes/Chords per Measure) ---
            tracks_data = {}
            
            # 1. Global Merged Chords (Time-Slice Analysis)
            tracks_data["Global_Merged_Chords"] = {}
            current_time_sig = None
            current_key_sig = None
            
            chordified_measures = []
            for measure in chordified.getElementsByClass('Measure'):
                m_num = measure.number
                if start_measure is not None and m_num < start_measure:
                    continue
                if end_measure is not None and m_num > end_measure:
                    continue
                chordified_measures.append(measure)

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
                    if self.note_format_config.show_chord_name:
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
            for i, part in enumerate(tqdm(parts, desc="Analyzing Instruments", leave=False)):
                base_part_name = part.partName or f"Unknown_Instrument_{i+1}"
                
                part_name = base_part_name
                suffix = 1
                while part_name in tracks_data:
                    part_name = f"{base_part_name}_{suffix}"
                    suffix += 1
                    
                tracks_data[part_name] = {}
                
                measures = []
                for measure in part.getElementsByClass('Measure'):
                    m_num = measure.number
                    if start_measure is not None and m_num < start_measure:
                        continue
                    if end_measure is not None and m_num > end_measure:
                        continue
                    measures.append(measure)
                for measure in measures:
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
            
            # Filter output
            final_tracks_data = {}
            if self.show_global_merged_chords and "Global_Merged_Chords" in tracks_data:
                final_tracks_data["Global_Merged_Chords"] = tracks_data["Global_Merged_Chords"]
            
            if self.show_individual_instruments:
                for k, v in tracks_data.items():
                    if k != "Global_Merged_Chords":
                        final_tracks_data[k] = v
            
            return {
                "estimated_key": str(key_sig),
                "mode": key_sig.mode,
                "time_signature": ts,
                "bpm": bpm,
                "tempo_map": tempo_map,
                "tempo_changes": tempo_changes_desc,
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
            print(f"[!] music21 Parsing Failed: {e}")
            return {"error": str(e)}
