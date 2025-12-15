from pathlib import Path
import json

from backend.pipeline.transcribe import transcribe
from backend.pipeline.config import PIANO_61KEY_CONFIG

AUDIO = r"C:\Code\music-note\inputs\my_song.wav"  # or .mp3

out_dir = Path("outputs") / Path(AUDIO).stem
out_dir.mkdir(parents=True, exist_ok=True)

result = transcribe(AUDIO, config=PIANO_61KEY_CONFIG)

# Dump whatever TranscriptionResult contains (safe introspection)
(out_dir / "result_debug.json").write_text(
    json.dumps(getattr(result, "__dict__", {"repr": repr(result)}), indent=2, default=str),
    encoding="utf-8",
)
print("Wrote:", out_dir / "result_debug.json")

# If your result already contains these fields, export them (no guessing beyond hasattr)
if hasattr(result, "musicxml"):
    (out_dir / "score.musicxml").write_text(result.musicxml, encoding="utf-8")
    print("Wrote:", out_dir / "score.musicxml")

if hasattr(result, "predicted_notes"):
    (out_dir / "predicted_notes.json").write_text(json.dumps(result.predicted_notes, indent=2, default=str), encoding="utf-8")
    print("Wrote:", out_dir / "predicted_notes.json")

if hasattr(result, "timeline"):
    (out_dir / "timeline.json").write_text(json.dumps(result.timeline, indent=2, default=str), encoding="utf-8")
    print("Wrote:", out_dir / "timeline.json")
