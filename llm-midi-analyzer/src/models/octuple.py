"""
octuple.py — OctupleMIDI Feature Extractors
============================================
Provides two strategies (Strategy Pattern / Open-Closed):

  MidiTok5DExtractor   → uses miditok.Octuple → 5-dim token IDs
  Octuple8DExtractor   → manual extraction    → 8-dim raw values
                         (Bar, Pos, Program, Pitch, Duration, Vel, TimeSig, Tempo)

Both return np.ndarray of shape (N, D) where D is the feature dimension.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import warnings
import numpy as np


# ────────────────────────────────────────────────────────────────
# Abstract base (Interface Segregation / Dependency Inversion)
# ────────────────────────────────────────────────────────────────
class IMidiExtractor(ABC):
    @abstractmethod
    def extract(self, midi_path: str) -> np.ndarray | None:
        """Return (N, D) array or None on failure."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


# ────────────────────────────────────────────────────────────────
# 5-dim: delegate to miditok.Octuple
# ────────────────────────────────────────────────────────────────
class MidiTok5DExtractor(IMidiExtractor):
    """
    Uses miditok.Octuple. Each note → (Pitch, Pos, Bar, Vel, Dur) token IDs.
    Integer vocabulary indices, not raw MIDI values.
    """
    @property
    def dim(self) -> int: return 5

    def __init__(self):
        from miditok import Octuple
        # Silence the "Attribute controls not compatible" warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self._tok = Octuple()

    def extract(self, midi_path: str) -> np.ndarray | None:
        try:
            from symusic import Score
            score = Score(midi_path)
            rows = []
            # Suppress bar-clipping UserWarning (expected for long MIDI files)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                for seq in self._tok(score):
                    ids = np.array(seq.ids, dtype=np.int16)   # (N, 5)
                    rows.append(ids)
            return np.concatenate(rows, axis=0) if rows else None
        except Exception:
            return None


# ────────────────────────────────────────────────────────────────
# 8-dim: manual extraction (full OctupleMIDI spec)
# ────────────────────────────────────────────────────────────────
class Octuple8DExtractor(IMidiExtractor):
    """
    Manually builds 8-dim feature tuples following the OctupleMIDI paper:
      (Bar, Pos, Program, Pitch, Duration, Velocity, TimeSig, Tempo)

    Values are discretized raw MIDI values (not vocab IDs).
    - Bar         : bar index (0-indexed)
    - Pos         : 32nd-note grid position within bar (0–31)
    - Program     : MIDI program 0–127 (drums excluded)
    - Pitch       : MIDI pitch 0–127
    - Duration    : duration in 32nd-note ticks (clamped 1–64)
    - Velocity    : quantized to 32 levels (0–31)
    - TimeSig     : numerator of time signature (e.g. 4 for 4/4)
    - Tempo       : BPM bucket 0–127 (mapped from ~30–350 BPM range)
    """
    @property
    def dim(self) -> int: return 8

    def extract(self, midi_path: str) -> np.ndarray | None:
        try:
            import pretty_midi
            # Suppress invalid MIDI type RuntimeWarning
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                pm = pretty_midi.PrettyMIDI(midi_path)
            pm.remove_invalid_notes()

            # ── Time signature ────────────────────────────────────
            ts_num = pm.time_signature_changes[0].numerator if pm.time_signature_changes else 4
            ts_num = max(1, min(ts_num, 127))

            # ── Tempo map ─────────────────────────────────────────
            try:
                tc_times, tc_bpms = pm.get_tempo_changes()
                tc_times = list(tc_times)
                tc_bpms = list(tc_bpms)
            except Exception:
                tc_times, tc_bpms = [0.0], [120.0]
            if not tc_times:
                tc_times, tc_bpms = [0.0], [120.0]

            def get_tempo_bucket(t: float) -> int:
                idx = max(0, int(np.searchsorted(tc_times, t, side="right")) - 1)
                bpm = tc_bpms[idx]
                return int(np.clip((bpm - 30.0) / (350.0 - 30.0) * 127, 0, 127))

            # ── Bar / grid geometry ───────────────────────────────
            spb_at0 = 60.0 / tc_bpms[0]
            bar_dur = ts_num * spb_at0
            grid = bar_dur / 32.0
            if grid <= 0: return None

            # ── Per-note encoding ─────────────────────────────────
            rows: list[np.ndarray] = []
            for inst in pm.instruments:
                if inst.is_drum: continue
                prog = int(np.clip(inst.program, 0, 127))
                for note in inst.notes:
                    bar_idx = int(note.start / bar_dur)
                    pos = int((note.start % bar_dur) / grid) % 32
                    dur = max(1, min(int((note.end - note.start) / grid), 64))
                    vel = int(np.clip(note.velocity // 4, 0, 31))
                    tempo_bucket = get_tempo_bucket(note.start)

                    rows.append(np.array([
                        bar_idx, pos, prog, note.pitch, dur, vel, ts_num, tempo_bucket,
                    ], dtype=np.int16))

            if not rows: return None
            return np.stack(rows, axis=0)
        except Exception as e:
            print(f"Exception extracting {midi_path}: {e}")
            return None

def octuple8d_to_midi(data: np.ndarray, out_path: str) -> None:
    """
    Reconstructs a MIDI file from a (N, 8) array of OctupleMIDI features.
    Features: (Bar, Pos, Program, Pitch, Duration, Velocity, TimeSig, Tempo)
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    
    # Simple heuristic: use the first note's TimeSig and Tempo for the whole sequence
    # In a real system, these would be global parameters or change over time.
    if len(data) == 0: return
    
    ts_num = int(data[0, 6])
    tempo_bucket = int(data[0, 7])
    bpm = 30.0 + (tempo_bucket / 127.0) * (350.0 - 30.0)
    
    spb = 60.0 / bpm
    bar_dur = ts_num * spb
    grid = bar_dur / 32.0
    
    # Group by program
    instruments = {}
    
    for row in data:
        # Clip for robustness against neural network noise
        bar_idx = int(max(0, row[0]))
        pos = int(np.clip(row[1], 0, 31))
        prog = int(np.clip(row[2], 0, 127))
        pitch = int(np.clip(row[3], 0, 127))
        dur = int(max(1, row[4])) # Guarantee end > start
        vel = int(np.clip(row[5], 0, 31))
        
        if prog not in instruments:
            inst = pretty_midi.Instrument(program=prog)
            pm.instruments.append(inst)
            instruments[prog] = inst
        
        start_time = (bar_idx * bar_dur) + (pos * grid)
        end_time = start_time + (dur * grid)
        
        note = pretty_midi.Note(
            velocity=int(vel * 4),
            pitch=int(pitch),
            start=start_time,
            end=end_time
        )
        instruments[prog].notes.append(note)
    
    pm.write(out_path)


# ────────────────────────────────────────────────────────────────
# Factory function (Open/Closed — add new modes here only)
# ────────────────────────────────────────────────────────────────
def get_extractor(token_mode: str) -> IMidiExtractor:
    if token_mode == "miditok_5d":
        return MidiTok5DExtractor()
    elif token_mode == "octuple_8d":
        return Octuple8DExtractor()
    else:
        raise ValueError(f"Unknown token_mode: {token_mode!r}. Choose 'miditok_5d' or 'octuple_8d'.")
