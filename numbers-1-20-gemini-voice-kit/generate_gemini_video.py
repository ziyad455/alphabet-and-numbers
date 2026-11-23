#!/usr/bin/env python3
"""Replace the spoken soundtrack in the provided 1–20 teaching video.

Requires: Python 3.10+, numpy, soundfile, ffmpeg, an internet-connected machine,
          and an API key in GEMINI_API_KEY (never stored in this project).

Uses Gemini's documented v1beta Interactions API with the current TTS model.
Video cue times were reviewed against the original 152.18 s video frames.
Fine-adjust CUES below if your preferred voice needs a slightly earlier onset.
"""

import argparse
import base64
import json
import getpass
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import soundfile as sf

VIDEO_DURATION = 152.183
API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/interactions'
NUMBER_WORDS = [
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
]
# Times are seconds from the START of the original video, aligned to the
# large numerals' entrance. Recheck while watching the final export.
NUMBER_TIMES = [10.7, 14.7, 18.7, 22.7, 26.7, 31.7, 37.7, 42.7, 46.7,
                51.7, 58.7, 64.7, 69.7, 75.7, 82.7, 87.7, 94.5,
                99.7, 107.7, 114.7]

# Each cue: (start, cached_clip_key, words, max_length_seconds)
CUES = [(3.1, 'intro', 'Numbers one to twenty. Listen and repeat.', 4.6)]
CUES += [(start, f'number-{i:02d}', word, 2.6)
         for i, (start, word) in enumerate(zip(NUMBER_TIMES, NUMBER_WORDS), 1)]
# Visual spelling emphasis stretches the 14 and 17 sections to ~10 seconds.
CUES += [(79.0, 'number-14', 'fourteen', 2.0),
         (97.0, 'number-17', 'seventeen', 1.7)]
CUES += [(121.5, 'lets-count', "Let's count!", 1.5)]
CUES += [(123.2 + i * 0.94, f'number-{i+1:02d}', word, 0.8)
         for i, word in enumerate(NUMBER_WORDS)]
CUES += [(144.0, 'outro', 'Great job! Now you try counting from one to twenty.', 7.8)]
CUES.sort(key=lambda x: x[0])


def run(args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def generate_one(key, text, path, api_key, model, voice):
    style = ('A friendly professional British English teacher, warm and clear, '
             'articulate exact words only, crisp consonants, patient beginner-level '
             'pace, natural intonation, no music or sound effects.')
    if key.startswith('number-'):
        style += ' Pronounce this single number distinctly with no introduction or repetition.'
    payload = {
        'model': model,
        'input': [{'type': 'user_input', 'content': [{
            'type': 'text', 'text': text,
            'annotations': [{'type': 'speech_metadata', 'style': style}]
        }]}],
        'response_format': {'type': 'audio'},
        'generation_config': {'speech_config': [{'voice': voice}]},
    }
    body = json.dumps(payload).encode('utf-8')
    last_exc = None
    for attempt in range(6):
        request = urllib.request.Request(
            API_ENDPOINT, data=body, method='POST',
            headers={'Content-Type': 'application/json', 'x-goog-api-key': api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=100) as response:
                data = json.load(response)
            audio = data.get('output_audio')
            encoded = audio.get('data') if isinstance(audio, dict) else None
            if not encoded:
                for step in data.get('steps', []):
                    if step.get('type') != 'model_output':
                        continue
                    for part in step.get('content', []):
                        if part.get('type') == 'audio' and part.get('data'):
                            encoded = part['data']
            if not encoded:
                raise RuntimeError(f'Gemini returned no audio for {key}; response keys: {list(data)}')
            wav_bytes = base64.b64decode(encoded)
            if not wav_bytes.startswith(b'RIFF'):
                raise RuntimeError(f'Unexpected audio format for {key}. Expected WAV from unary request.')
            path.write_bytes(wav_bytes)
            return
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504):
                # Do not include authentication headers or token in messages.
                raise RuntimeError(f'Gemini HTTP {exc.code} on {key}; check key, model, and quota.') from None
        except urllib.error.URLError as exc:
            last_exc = exc
        if attempt < 5:
            retry_after = last_exc.headers.get('Retry-After') if isinstance(last_exc, urllib.error.HTTPError) else None
            try:
                advised_delay = float(retry_after) if retry_after else 0
            except ValueError:
                advised_delay = 0
            delay = min(60, max(advised_delay, min(60, 15 * 2 ** attempt))) if isinstance(last_exc, urllib.error.HTTPError) and last_exc.code == 429 else min(30, 2 ** attempt + 1)
            print(f'Gemini request for {key} will retry after {delay:.0f}s (HTTP {last_exc.code if isinstance(last_exc, urllib.error.HTTPError) else "network error"}).', flush=True)
            time.sleep(delay)
    reason = f'HTTP {last_exc.code}' if isinstance(last_exc, urllib.error.HTTPError) else type(last_exc).__name__
    raise RuntimeError(f'Gemini audio generation failed for {key} after retries: {reason}')


def load_trimmed_audio(path, work_dir):
    wav48 = work_dir / ('decoded-' + path.stem + '.wav')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(path),
         '-ac', '1', '-ar', '48000', '-c:a', 'pcm_f32le', str(wav48)])
    audio, sample_rate = sf.read(wav48, dtype='float32')
    if sample_rate != 48000:
        raise RuntimeError('FFmpeg did not produce 48k audio')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0
    if peak < 0.0005:
        raise RuntimeError(f'Silent generated clip: {path}')
    threshold = max(0.0015, peak * 0.045)
    active = np.flatnonzero(np.abs(audio) > threshold)
    if len(active):
        start = max(0, int(active[0] - 0.035 * sample_rate))
        end = min(len(audio), int(active[-1] + 0.09 * sample_rate))
        audio = audio[start:end]
    wav48.unlink(missing_ok=True)
    return audio


def normalize_clip(samples):
    peak = float(np.max(np.abs(samples)))
    active = samples[np.abs(samples) > max(0.01, peak * 0.08)]
    if not len(active):
        return samples
    rms = float(np.sqrt(np.mean(active ** 2)))
    gain = min(0.12 / rms, 0.85 / peak)
    return samples * gain


def fit_to_max_duration(samples, max_seconds, work_dir, index):
    rate = 48000
    max_samples = int(max_seconds * rate)
    if len(samples) <= max_samples:
        return samples
    factor = len(samples) / max_samples * 1.015
    # Avoid extreme pitch-preserving time compression of long unexpected speech.
    if factor > 1.9:
        raise RuntimeError(f'Cue {index}: speech {len(samples)/rate:.2f}s is too long for {max_seconds:.2f}s slot.')
    from_wav = work_dir / f'fit-from-{index}.wav'
    to_wav = work_dir / f'fit-to-{index}.wav'
    sf.write(from_wav, samples, rate)
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(from_wav),
         '-af', f'atempo={factor:.5f}', '-ar', str(rate), str(to_wav)])
    result, _ = sf.read(to_wav, dtype='float32')
    from_wav.unlink(missing_ok=True)
    to_wav.unlink(missing_ok=True)
    if len(result) > max_samples:
        result = result[:max_samples]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default=str(Path(__file__).resolve().parent.parent / 'chapter-1/slide-8/assets/numbers.mp4'), help='Input original video')
    parser.add_argument('--output', default='numbers-1-20-gemini-voice.mp4', help='Final MP4')
    parser.add_argument('--work-dir', default='generated-voice', help='Cache and intermediate audio')
    parser.add_argument('--model', default='gemini-3.8-flash-tts')
    parser.add_argument('--voice', default='Kore')
    parser.add_argument('--use-existing-clips', action='store_true',
                        help='Mix already-generated WAV clips; do not call Gemini API')
    parser.add_argument('--print-cues', action='store_true', help='Show narration timeline only')
    args = parser.parse_args()
    for onset, _, words, limit in CUES:
        if args.print_cues:
            print(f'{onset:7.2f}s | {limit:4.2f}s | {words}')
    if args.print_cues:
        return
    video = Path(args.video)
    if not video.exists():
        parser.error(f'Video not found: {video}')
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    clips = work / 'clips'
    clips.mkdir(exist_ok=True)
    needed = {key: txt for _, key, txt, _ in CUES}
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not args.use_existing_clips and not api_key:
        if not sys.stdin.isatty():
            parser.error('Set GEMINI_API_KEY in your environment before running non-interactively.')
        api_key = getpass.getpass('Gemini API key (hidden; not saved): ').strip()
        if not api_key:
            parser.error('A Gemini API key is required to generate new speech.')
    for i, (key, text) in enumerate(needed.items(), 1):
        target = clips / f'{key}.wav'
        if target.exists() and target.stat().st_size > 256:
            print(f'[{i:02d}/{len(needed)}] Cached: {key}')
            continue
        if args.use_existing_clips:
            raise RuntimeError(f'Missing clip {target}')
        print(f'[{i:02d}/{len(needed)}] Generating: {key} ({text})', flush=True)
        generate_one(key, text, target, api_key, args.model, args.voice)
        time.sleep(12)  # Leave room for low per-minute TTS quotas.

    audio = np.zeros(int(math.ceil(VIDEO_DURATION * 48000)), dtype=np.float32)
    decoded_cache = {}
    for index, (start, key, words, max_duration) in enumerate(CUES):
        if key not in decoded_cache:
            decoded_cache[key] = normalize_clip(load_trimmed_audio(clips / (key + '.wav'), work))
        wav = fit_to_max_duration(decoded_cache[key], max_duration, work, index)
        wav = wav.copy()
        fade = min(480, len(wav) // 8)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            wav[:fade] *= ramp
            wav[-fade:] *= ramp[::-1]
        offset = round(start * 48000)
        length = min(len(wav), len(audio) - offset)
        if length > 0:
            audio[offset:offset + length] += wav[:length]
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= min(0.92 / peak, 1.75)
    sound = work / 'narration-synced.wav'
    sf.write(sound, audio, 48000, subtype='PCM_16')
    print(f'Wrote synchronized voice track: {sound}')
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
         '-i', str(video), '-i', str(sound), '-map', '0:v:0', '-map', '1:a:0',
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-t', str(VIDEO_DURATION),
         '-movflags', '+faststart', str(args.output)])
    print(f'Wrote final video: {args.output}')
    print('IMPORTANT: Watch and listen to the entire result; adjust cue times if necessary.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
