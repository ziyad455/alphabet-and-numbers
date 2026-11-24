"""Build the Numbers 1–20 soundtrack from timed ElevenLabs narration clips.

The existing MP4 supplies its original video stream only. Cached WAV clips in
assets/audio are ElevenLabs recordings; missing clips are generated with Alice.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import requests
import soundfile as sf


ROOT = Path(__file__).resolve().parents[3]
SLIDE = ROOT / "chapter-1/slide-8"
VIDEO = SLIDE / "assets/numbers-1-20-elevenlabs-voice.mp4"
CLIPS = SLIDE / "assets/audio"
MANIFEST = Path(__file__).with_name("cue_sheet.json")
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"  # Alice, the Chapter 2 numbers narrator
MODEL_ID = "eleven_multilingual_v2"
SAMPLE_RATE = 48_000
DURATION = 152.183
FFMPEG = os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"]
APPEARANCES = [10.7, 14.7, 18.7, 22.7, 26.7, 31.7, 37.7, 42.7, 46.7, 51.7,
               58.7, 64.7, 69.7, 75.7, 82.7, 87.7, 94.5, 99.7, 107.7, 114.7]


def cues():
    result = []

    def add(section, cue_id, text, start, max_duration, clip):
        result.append({"id": cue_id, "section": section, "text": text,
                       "start_seconds": start, "max_duration_seconds": max_duration,
                       "audio_file": clip + ".wav"})

    add("introduction", "intro", "Numbers one to twenty. Listen and repeat.", 3.1, 4.6, "intro-v2")
    for number, (word, start) in enumerate(zip(WORDS, APPEARANCES), 1):
        add("individual_numbers", f"number_{number}", word, start, 2.6, f"number-{number:02d}")
    add("counting", "lets_count", "Let's count!", 121.5, 1.5, "lets-count-v2")
    for number, word in enumerate(WORDS, 1):
        add("counting", f"count_{number}", word, 123.2 + (number - 1) * .94,
            .8, f"number-{number:02d}")
    add("ending", "outro", "Great job! Now you try counting from one to twenty.",
        144.0, 7.8, "outro-v2")
    return result


def api_key():
    key = os.getenv("ELEVENLABS_API_KEY")
    if key:
        return key
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            if name.strip() in ("ELEVENLABS_API_KEY", "api"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError("ElevenLabs API key is unavailable")


def ffmpeg(*args):
    if not FFMPEG:
        raise RuntimeError("FFmpeg is required; set FFMPEG_BINARY")
    result = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                             *map(str, args)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:500])


def ensure_clips(cue_list):
    key = None
    count = 0
    session = requests.Session()
    for cue in cue_list:
        target = CLIPS / cue["audio_file"]
        if target.exists() and target.stat().st_size > 1000:
            continue
        if key is None:
            key = api_key()
        for attempt in range(4):
            response = session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": key, "Content-Type": "application/json"},
                json={"text": cue["text"], "model_id": MODEL_ID,
                      "voice_settings": {"stability": .65, "similarity_boost": .8, "speed": 1.0}},
                timeout=90)
            count += 1
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt + 1)
                continue
            if not response.ok or not response.headers.get("content-type", "").startswith("audio/"):
                raise RuntimeError(f"ElevenLabs generation failed for {cue['id']}: HTTP {response.status_code}")
            temporary = CLIPS / (target.stem + ".mp3")
            temporary.write_bytes(response.content)
            ffmpeg("-i", temporary, "-ac", "1", "-ar", SAMPLE_RATE, target)
            temporary.unlink()
            print(f"Generated {cue['id']} with ElevenLabs", flush=True)
            break
    return count


def fitted_clip(path, max_duration, work_dir, cue_id):
    clip, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE or clip.ndim != 1 or not len(clip):
        raise RuntimeError(f"Invalid narration clip: {path.name}")
    peak = float(np.max(np.abs(clip)))
    if peak < .001:
        raise RuntimeError(f"Silent narration clip: {path.name}")
    active = np.flatnonzero(np.abs(clip) > max(.0015, peak * .045))
    clip = clip[max(0, active[0] - 1680):min(len(clip), active[-1] + 4320)].copy()
    original_duration = len(clip) / rate
    tempo = 1.0
    if original_duration > max_duration:
        tempo = original_duration / max_duration * 1.015
        if tempo > 1.5:
            raise RuntimeError(f"Cue {cue_id} cannot fit its visual slot naturally")
        source = work_dir / f"{cue_id}-source.wav"
        output = work_dir / f"{cue_id}-fitted.wav"
        sf.write(source, clip, rate)
        ffmpeg("-i", source, "-af", f"atempo={tempo:.5f}", "-ar", rate, output)
        clip, _ = sf.read(output, dtype="float32")
    rms = float(np.sqrt(np.mean(clip * clip)))
    clip *= min(.12 / rms, .85 / float(np.max(np.abs(clip))))
    fade = min(480, len(clip) // 5)
    clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return clip, original_duration, tempo


def main():
    if not VIDEO.exists():
        raise RuntimeError(f"Source video missing: {VIDEO}")
    work_dir = Path("/tmp/ch1-8-elevenlabs-work")
    work_dir.mkdir(parents=True, exist_ok=True)
    cue_list = cues()
    requests_made = ensure_clips(cue_list)
    soundtrack = np.zeros(math.ceil(DURATION * SAMPLE_RATE), dtype=np.float32)
    for cue in cue_list:
        clip, original_duration, tempo = fitted_clip(
            CLIPS / cue["audio_file"], cue["max_duration_seconds"], work_dir, cue["id"])
        start = round(cue["start_seconds"] * SAMPLE_RATE)
        end = start + len(clip)
        if end > len(soundtrack):
            raise RuntimeError(f"Cue {cue['id']} exceeds video duration")
        if np.any(soundtrack[start:end]):
            raise RuntimeError(f"Cue {cue['id']} overlaps another narration cue")
        soundtrack[start:end] = clip
        cue["original_duration_seconds"] = round(original_duration, 3)
        cue["final_duration_seconds"] = round(len(clip) / SAMPLE_RATE, 3)
        cue["tempo_adjustment"] = round(tempo, 4)
    track = work_dir / "numbers-1-20-narration.wav"
    sf.write(track, soundtrack, SAMPLE_RATE)
    output = VIDEO.with_name("numbers-1-20-elevenlabs-voice.tmp.mp4")
    ffmpeg("-i", VIDEO, "-i", track, "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", SAMPLE_RATE,
           "-t", DURATION, "-movflags", "+faststart", output)
    output.replace(VIDEO)
    MANIFEST.write_text(json.dumps({"video_duration_seconds": DURATION, "voice": "Alice",
                                    "voice_id": VOICE_ID, "model_id": MODEL_ID,
                                    "api_requests_this_run": requests_made,
                                    "cues": cue_list}, indent=2) + "\n")
    print(f"Finished {len(cue_list)} cues; {requests_made} new ElevenLabs requests: {VIDEO}")


if __name__ == "__main__":
    main()
