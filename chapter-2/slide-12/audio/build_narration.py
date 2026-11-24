"""Build the ElevenLabs soundtrack for the existing Numbers 20–50 video.

Run from the repository root with the project virtual environment. Generated
clips are cached outside the project by default; only the final MP4 and cue
sheet belong in the slide. The original M4A is never read by this script or
used in the export.
"""

from __future__ import annotations

import argparse
import hashlib
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
SLIDE = ROOT / "chapter-2/slide-12"
SOURCE_VIDEO = SLIDE / "assets/numbers-20-50.mp4"
OUTPUT_VIDEO = SLIDE / "assets/numbers-20-50-elevenlabs.mp4"
MANIFEST = Path(__file__).with_name("cue_sheet.json")
VOICE_ID = "Xb7hH8MSUJpSbSDYk0k2"  # Alice: British English educator
MODEL_ID = "eleven_multilingual_v2"
SAMPLE_RATE = 48_000
FFMPEG = Path(os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg") or "/tmp/bec-gemini-venv/bin/ffmpeg")


def cues() -> list[dict]:
    result = []

    def add(section, cue_id, spoken, start, end, visual_start=None, visual_end=None, speed=1.0, note=""):
        result.append({
            "id": cue_id, "section": section, "text": spoken,
            "visual_start_seconds": start if visual_start is None else visual_start,
            "visual_end_seconds": end if visual_end is None else visual_end,
            "start_seconds": start, "end_seconds": end,
            "max_duration_seconds": round(end - start, 3),
            "speed": speed, "note": note,
        })

    add("introduction", "hello", "Hello.", .35, 1.38)
    add("introduction", "dotty", "I'm Dotty.", 1.65, 2.90)
    add("introduction", "lesson_intro", "Let's learn numbers twenty to fifty.", 3.02, 6.40)
    add("the_tens", "tens_intro", "First, the tens.", 7.70, 9.55)
    for word, first, second in (
        ("Twenty", 10.17, 11.77), ("Thirty", 13.57, 15.07),
        ("Forty", 16.77, 18.37), ("Fifty", 20.37, 21.67),
    ):
        add("the_tens", f"tens_{word.lower()}_1", f"{word}.", first, second - .15,
            visual_start=first - .1, visual_end=second)

    add("teen_or_ty", "listen", "Listen.", 23.50, 24.18)
    add("teen_or_ty", "teen_or_ty", "Teen or ty?", 24.72, 26.05,
        note="The video heading uses -teen and -ty.")
    for cue_id, word, start, end in (
        ("thirteen", "Thirteen", 27.06, 28.00),
        ("thirty", "Thirty", 28.42, 29.58),
        ("fourteen", "Fourteen", 30.67, 31.72),
        ("forty", "Forty", 31.98, 33.16),
        ("fifteen", "Fifteen", 34.26, 35.33),
        ("fifty", "Fifty", 35.66, 36.88),
    ):
        add("teen_or_ty", cue_id, f"{word}.", start, end,
            note="Keep the teen/ty stress contrast clear.")

    add("make_numbers", "make_intro", "Now, let's make numbers.", 38.40, 40.14)
    for n, expression, answer, expression_start, expression_end, answer_start, answer_end in (
        (1, "Twenty plus one.", "Twenty-one.", 40.70, 42.62, 43.20, 44.34),
        (2, "Twenty plus five.", "Twenty-five.", 45.05, 47.54, 48.08, 48.96),
        (3, "Thirty plus four.", "Thirty-four.", 49.06, 52.42, 53.08, 53.96),
        (4, "Forty plus seven.", "Forty-seven.", 54.04, 57.46, 58.06, 58.96),
    ):
        add("make_numbers", f"expression_{n}", expression, expression_start, expression_end)
        add("make_numbers", f"answer_{n}", answer, answer_start, answer_end,
            note="Speak only after the visual answer appears.")

    add("counting", "count_intro", "Count with me.", 59.87, 60.94)
    appearances = [
        61.30, 62.50, 63.60, 64.80, 65.90, 67.10, 68.20, 69.30,
        70.50, 71.60, 72.80, 73.90, 75.00, 76.20, 77.30, 78.50,
        79.60, 80.80, 81.90, 83.00, 84.20, 85.30, 86.50, 87.60,
        88.70, 89.90, 91.00, 92.20, 93.30, 94.40, 95.60,
    ]
    assert len(appearances) == 31
    for index, visual_start in enumerate(appearances):
        number = index + 20
        visual_end = appearances[index + 1] if index < 30 else 96.83
        add("counting", f"count_{number}", f"{number_word(number)}.",
            visual_start + .06, visual_end - .08,
            visual_start=visual_start, visual_end=visual_end, speed=1.08,
            note="Tile appearance measured from the source video at 0.1-second intervals.")

    add("quick_quiz", "quiz_intro", "Quick quiz.", 98.72, 99.66)
    add("quick_quiz", "quiz_question", "What number is it?", 100.18, 101.52)
    for n, answer, start, end in (
        (1, "Twenty-six.", 104.98, 106.37),
        (2, "Thirty-eight.", 110.26, 111.50),
        (3, "Forty-three.", 115.20, 116.56),
    ):
        add("quick_quiz", f"quiz_answer_{n}", answer, start, end,
            visual_start=start - .06, visual_end=end,
            note="Answer narration starts after the visual reveal.")

    add("ending", "great_job", "Great job!", 117.51, 118.62)
    add("ending", "can_count", "Now you can count to fifty.", 119.28, 121.50)
    add("ending", "bye_bye", "Bye-bye.", 121.96, 123.13)
    return result


def number_word(number: int) -> str:
    tens = {20: "Twenty", 30: "Thirty", 40: "Forty", 50: "Fifty"}
    if number in tens:
        return tens[number]
    ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    return f"{tens[(number // 10) * 10]}-{ones[number % 10]}"


def api_key() -> str:
    key = os.getenv("ELEVENLABS_API_KEY")
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip() in ("ELEVENLABS_API_KEY", "api"):
                return value.strip().strip('"').strip("'")
    raise RuntimeError("ElevenLabs key is missing from the environment")


def run_ffmpeg(*arguments):
    completed = subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments)],
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip()[:500])


def generate_clips(cue_list, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = api_key()
    session = requests.Session()
    requests_made = 0
    for cue in cue_list:
        settings = {"stability": .65, "similarity_boost": .8, "speed": cue["speed"]}
        identity = json.dumps([cue["text"], VOICE_ID, MODEL_ID, settings], sort_keys=True)
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        audio_file = cache_dir / f"{digest}.mp3"
        cue["audio_file"] = str(audio_file)
        if audio_file.exists() and audio_file.stat().st_size > 1000:
            continue
        for attempt in range(4):
            try:
                response = session.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
                    params={"output_format": "mp3_44100_128"},
                    headers={"xi-api-key": key, "Content-Type": "application/json"},
                    json={"text": cue["text"], "model_id": MODEL_ID,
                          "voice_settings": settings}, timeout=90)
                requests_made += 1
            except requests.RequestException as exc:
                if attempt == 3:
                    raise RuntimeError(f"TTS failed for {cue['id']}: {exc}") from exc
                time.sleep(2 ** attempt)
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2 ** attempt + 1)
                continue
            if not response.ok or not response.headers.get("content-type", "").startswith("audio/"):
                raise RuntimeError(f"TTS failed for {cue['id']}: HTTP {response.status_code} {response.text[:250]}")
            audio_file.write_bytes(response.content)
            print(f"generated {cue['id']} ({len(response.content)} bytes)", flush=True)
            break
    return requests_made


def read_trimmed_clip(path: Path, work_dir: Path):
    wav_file = work_dir / (path.stem + ".wav")
    if not wav_file.exists():
        run_ffmpeg("-i", path, "-ac", 1, "-ar", SAMPLE_RATE, wav_file)
    samples, sample_rate = sf.read(wav_file, dtype="float32")
    assert sample_rate == SAMPLE_RATE and len(samples)
    frame = 480  # 10 ms
    window = np.array([np.sqrt(np.mean(chunk * chunk)) for chunk in
                       (samples[i:i + frame] for i in range(0, len(samples), frame))])
    threshold = max(.005, float(window.max()) * .028)
    active = np.flatnonzero(window > threshold)
    if len(active) == 0:
        raise RuntimeError(f"No speech detected in {path.name}")
    beginning = max(0, (int(active[0]) - 2) * frame)
    ending = min(len(samples), (int(active[-1]) + 3) * frame)
    return samples[beginning:ending]


def assemble(cue_list, cache_dir, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = 124.23
    soundtrack = np.zeros(math.ceil(duration * SAMPLE_RATE), dtype=np.float32)
    clip_cache = {}
    previous_end = 0
    for cue in cue_list:
        path = Path(cue["audio_file"])
        if not path.exists():
            raise RuntimeError(f"Missing audio for {cue['id']}: {path}")
        if path not in clip_cache:
            clip_cache[path] = read_trimmed_clip(path, work_dir)
        clip = clip_cache[path].copy()
        original_duration = len(clip) / SAMPLE_RATE
        available = cue["end_seconds"] - cue["start_seconds"]
        tempo = 1.0
        if original_duration > available:
            tempo = original_duration / available * 1.01
            if tempo > 1.3:
                raise RuntimeError(f"Cue {cue['id']} needs {original_duration:.2f}s in {available:.2f}s; regenerate faster")
            fitted = work_dir / f"{cue['id']}-fit.wav"
            raw = work_dir / f"{cue['id']}-raw.wav"
            sf.write(raw, clip, SAMPLE_RATE)
            run_ffmpeg("-i", raw, "-af", f"atempo={tempo:.4f}", "-ar", SAMPLE_RATE, fitted)
            clip, _ = sf.read(fitted, dtype="float32")
        start = round(cue["start_seconds"] * SAMPLE_RATE)
        end = start + len(clip)
        if end > len(soundtrack):
            raise RuntimeError(f"Cue {cue['id']} extends beyond video")
        if start < previous_end:
            raise RuntimeError(f"Cue {cue['id']} overlaps the previous cue")
        rms = float(np.sqrt(np.mean(clip * clip)))
        if rms < .001:
            raise RuntimeError(f"Cue {cue['id']} is unexpectedly quiet")
        clip *= min(.15 / rms, .89 / float(np.max(np.abs(clip))))
        fade = min(480, len(clip) // 5)
        clip[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        clip[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        soundtrack[start:end] = clip
        previous_end = end
        cue["generated_duration_seconds"] = round(original_duration, 3)
        cue["final_duration_seconds"] = round(len(clip) / SAMPLE_RATE, 3)
        cue["tempo_adjustment"] = round(tempo, 4)
        if cue["final_duration_seconds"] > available + .025:
            raise RuntimeError(f"Cue {cue['id']} overflows its visual window")
    if np.max(np.abs(soundtrack)) > .98:
        raise RuntimeError("Soundtrack clips")
    soundtrack_file = work_dir / "elevenlabs-soundtrack.wav"
    sf.write(soundtrack_file, soundtrack, SAMPLE_RATE)
    run_ffmpeg("-i", SOURCE_VIDEO, "-i", soundtrack_file,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-ar", SAMPLE_RATE, "-t", duration,
               "-movflags", "+faststart", OUTPUT_VIDEO)
    return duration


def repair_existing(work_dir):
    """Remove the four redundant tens cues without regenerating unaffected speech."""
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    duplicate_ids = {f"tens_{word}_2" for word in ("twenty", "thirty", "forty", "fifty")}
    duplicates = [cue for cue in manifest["cues"] if cue["id"] in duplicate_ids]
    if len(duplicates) != 4:
        raise RuntimeError("Expected four duplicate tens cues in the existing manifest")
    track_file = work_dir / "current-soundtrack.wav"
    run_ffmpeg("-i", OUTPUT_VIDEO, "-map", "0:a:0", "-ac", "1", "-ar", SAMPLE_RATE, track_file)
    samples, sample_rate = sf.read(track_file, dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise RuntimeError("Unexpected soundtrack sample rate")
    for cue in duplicates:
        beginning = round(cue["start_seconds"] * SAMPLE_RATE)
        ending = beginning + round(cue["final_duration_seconds"] * SAMPLE_RATE)
        # Ten-millisecond ramps prevent clicks at the boundaries.
        fade = min(480, (ending - beginning) // 2)
        samples[beginning:beginning + fade] *= np.linspace(1, 0, fade, dtype=np.float32)
        samples[beginning + fade:ending - fade] = 0
        samples[ending - fade:ending] *= np.linspace(0, 1, fade, dtype=np.float32)
    repaired_track = work_dir / "repaired-soundtrack.wav"
    sf.write(repaired_track, samples, SAMPLE_RATE)
    repaired_video = OUTPUT_VIDEO.with_name("numbers-20-50-repaired.tmp.mp4")
    run_ffmpeg("-i", SOURCE_VIDEO, "-i", repaired_track,
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
               "-b:a", "192k", "-ar", SAMPLE_RATE, "-t", manifest["video_duration_seconds"],
               "-movflags", "+faststart", repaired_video)
    repaired_video.replace(OUTPUT_VIDEO)
    manifest["cues"] = [cue for cue in manifest["cues"] if cue["id"] not in duplicate_ids]
    manifest["repair"] = "Removed the second pronunciation within each tens presentation; all other cues retained."
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Repaired {len(duplicates)} duplicate cues: {OUTPUT_VIDEO}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ch2-12-elevenlabs-cache"))
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/ch2-12-elevenlabs-work"))
    parser.add_argument("--repair-existing", action="store_true",
                        help="Remove redundant tens cues from the current MP4 without API requests")
    arguments = parser.parse_args()
    if arguments.repair_existing:
        repair_existing(arguments.work_dir)
        return
    cue_list = cues()
    assert len({cue["id"] for cue in cue_list}) == len(cue_list)
    assert all(cue_list[i]["end_seconds"] <= cue_list[i + 1]["start_seconds"] for i in range(len(cue_list) - 1))
    assert SOURCE_VIDEO.exists()
    requests_made = generate_clips(cue_list, arguments.cache_dir)
    duration = assemble(cue_list, arguments.cache_dir, arguments.work_dir)
    portable_cues = [{**cue, "audio_file": Path(cue["audio_file"]).name} for cue in cue_list]
    MANIFEST.write_text(json.dumps({"video_duration_seconds": duration,
                                    "voice": "Alice", "voice_id": VOICE_ID,
                                    "model_id": MODEL_ID,
                                    "unique_audio_clips": len({cue["audio_file"] for cue in cue_list}),
                                    "api_requests_for_final_clips": len({cue["audio_file"] for cue in cue_list}),
                                    "api_requests_this_run": requests_made,
                                    "cues": portable_cues}, indent=2) + "\n")
    print(f"Finished {len(cue_list)} cues, {requests_made} requests: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
