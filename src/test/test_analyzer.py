import json
from app import AppConfig, StandardNoteFormatter, Music21MidiAnalyzer

config = AppConfig()
formatter = StandardNoteFormatter()
analyzer = Music21MidiAnalyzer(config, formatter)

data = analyzer.analyze_file("midi/no.12_v2.mid")
print(json.dumps(data["detailed_tracks"]["Piano"], indent=2))
