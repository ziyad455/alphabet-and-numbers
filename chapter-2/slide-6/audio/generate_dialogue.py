"""Generate the verified Ahmed and Sara dialogue with ElevenLabs.

Run from the repository root. Raw API responses stay in /tmp; only the twelve
finished lesson MP3s, transcript manifest, and browser data are written here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
SLIDE = ROOT / "chapter-2/slide-6"
ASSETS = SLIDE / "assets/audio"
SHARED = ROOT / "chapter-2/shared"
RAW_CACHE = Path("/tmp/ch2-dialogue-elevenlabs-raw")
FFMPEG = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg") or "/tmp/bec-gemini-venv/bin/ffmpeg"
MODEL = "eleven_multilingual_v2"
VOICES = {
    "ahmed": {"name": "George", "id": "JBFqnCBsd6RMkjVDRZzb"},
    "sara": {"name": "Lily", "id": "pFZP5JQG7iQjIQuC4Bku"},
}
SETTINGS = {"stability": 0.5, "similarity_boost": 0.8, "style": 0.12, "speed": 0.86}

# Checked against the source video's visible speech bubbles and its old cue sheet.
LINES = [
    ("D01", "ahmed", "Hello! I'm Ahmed.", 16.35, 18.60),
    ("D02", "sara", "Hi, Ahmed! I'm Sara.", 19.35, 22.60),
    ("D03", "ahmed", "Nice to meet you.", 22.85, 24.35),
    ("D04", "sara", "Nice to meet you, too.", 24.85, 27.10),
    ("D05", "ahmed", "Where are you from?", 27.60, 28.85),
    ("D06", "sara", "I'm from Egypt. And you?", 29.35, 32.35),
    ("D07", "ahmed", "I'm from Jordan.", 32.85, 34.35),
    ("D08", "sara", "What's your job?", 34.60, 35.85),
    ("D09", "ahmed", "I'm a teacher. And you?", 36.35, 38.85),
    ("D10", "sara", "I'm an engineer.", 39.35, 40.85),
    ("D11", "ahmed", "Goodbye, Sara!", 41.10, 42.85),
    ("D12", "sara", "Bye! See you!", 43.35, 45.35),
]
PORTRAITS = {
    "ahmed": "https://images.pexels.com/photos/6481907/pexels-photo-6481907.jpeg?auto=compress&cs=tinysrgb&w=800",
    "sara": "https://images.pexels.com/photos/4728683/pexels-photo-4728683.jpeg?auto=compress&cs=tinysrgb&w=800",
}
PHOTO_SOURCES = {
    "ahmed": "https://www.pexels.com/photo/man-in-blue-dress-shirt-6481907/",
    "sara": "https://www.pexels.com/photo/a-woman-wearing-a-hijab-4728683/",
}


def key_from_environment() -> str:
    if os.getenv("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"]
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() in ("ELEVENLABS_API_KEY", "api"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError("ElevenLabs API key is unavailable")


def ffmpeg(*args):
    result = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:500])


def generate_raw(session: requests.Session, key: str, record: tuple, index: int) -> Path:
    cue_id, speaker, spoken, _, _ = record
    voice = VOICES[speaker]
    previous = LINES[index - 1][2] if index else None
    following = LINES[index + 1][2] if index < len(LINES) - 1 else None
    settings = {**SETTINGS, "speed": 0.78 if cue_id in ("D04", "D12") else SETTINGS["speed"]}
    payload = {"text": spoken, "model_id": MODEL, "voice_settings": settings}
    if previous:
        payload["previous_text"] = previous
    if following:
        payload["next_text"] = following
    cache_key = hashlib.sha256(json.dumps([voice["id"], payload], sort_keys=True).encode()).hexdigest()[:16]
    path = RAW_CACHE / f"{cue_id.lower()}-{cache_key}.mp3"
    if path.exists() and path.stat().st_size > 1000:
        return path
    for attempt in range(4):
        try:
            response = session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice['id']}",
                headers={"xi-api-key": key, "Content-Type": "application/json"},
                params={"output_format": "mp3_44100_128"}, json=payload, timeout=90)
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"Audio generation failed for {cue_id}: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
            time.sleep(2 ** attempt + 1)
            continue
        if not response.ok or not response.headers.get("content-type", "").startswith("audio/"):
            raise RuntimeError(f"Audio generation failed for {cue_id}: HTTP {response.status_code} {response.text[:200]}")
        path.write_bytes(response.content)
        print(f"Generated {cue_id}: {speaker}", flush=True)
        return path
    raise RuntimeError(f"Audio generation failed for {cue_id}")


def clean_audio(raw: Path, finished: Path) -> float:
    wav = RAW_CACHE / f"{raw.stem}.wav"
    ffmpeg("-i", raw, "-ac", 1, "-ar", 48000, wav)
    samples, rate = sf.read(wav, dtype="float32")
    frame = 480
    energy = np.array([np.sqrt(np.mean(samples[i:i + frame] ** 2))
                       for i in range(0, len(samples), frame)])
    active = np.flatnonzero(energy > max(0.003, float(energy.max()) * 0.025))
    if not len(active):
        raise RuntimeError(f"Generated audio is silent: {raw.name}")
    samples = samples[max(0, (int(active[0]) - 4) * frame):
                      min(len(samples), (int(active[-1]) + 5) * frame)].copy()
    rms = float(np.sqrt(np.mean(samples ** 2)))
    samples *= min(0.13 / rms, 0.85 / float(np.max(np.abs(samples))))
    fade = min(480, len(samples) // 5)
    samples[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    samples[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    cleaned = RAW_CACHE / f"{raw.stem}-clean.wav"
    sf.write(cleaned, samples, rate)
    ffmpeg("-i", cleaned, "-codec:a", "libmp3lame", "-b:a", "128k", finished)
    return round(len(samples) / rate, 3)


def main():
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    key = key_from_environment()
    session = requests.Session()
    counts = {"ahmed": 0, "sara": 0}
    transcript = []
    for index, record in enumerate(LINES):
        cue_id, speaker, spoken, visual_start, visual_end = record
        counts[speaker] += 1
        filename = f"{speaker}-{counts[speaker]:02}.mp3"
        raw = generate_raw(session, key, record, index)
        duration = clean_audio(raw, ASSETS / filename)
        transcript.append({"id": cue_id, "speaker": speaker, "text": spoken,
                           "visual_start_seconds": visual_start,
                           "visual_end_seconds": visual_end,
                           "audio_file": filename,
                           "audio_duration_seconds": duration,
                           "voice_name": VOICES[speaker]["name"],
                           "voice_id": VOICES[speaker]["id"]})
    manifest = {"source_video": "introducing-yourself.mp4", "source_video_has_audio": False,
                "model": MODEL, "voices": VOICES, "photo_sources": PHOTO_SOURCES,
                "lines": transcript}
    (SLIDE / "audio/dialogue.json").write_text(json.dumps(manifest, indent=2) + "\n")
    browser_data = {"portraits": PORTRAITS,
                    "lines": [{"id": line["id"], "speaker": line["speaker"],
                               "text": line["text"], "audio": line["audio_file"]}
                              for line in transcript]}
    (SHARED / "dialogue-data.js").write_text("window.BEC_DIALOGUE = " + json.dumps(browser_data, indent=2) + ";\n")
    print(f"Completed {len(transcript)} dialogue files")


if __name__ == "__main__":
    main()
