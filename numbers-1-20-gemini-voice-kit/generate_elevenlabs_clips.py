#!/usr/bin/env python3
"""Generate one cached ElevenLabs voice clip per unique Numbers 1–20 cue.

Run from this directory with ELEVENLABS_API_KEY set. Then assemble with
generate_gemini_video.py --use-existing-clips --work-dir generated-voice/elevenlabs.
"""

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

from generate_gemini_video import CUES

VOICE_ID = 'Xb7hH8MSUJpSbSDYk0k2'  # Alice, British English educator
MODEL_ID = 'eleven_multilingual_v2'
CLIPS = Path('generated-voice/elevenlabs/clips')


def generate(text, target, api_key):
    request_body = json.dumps({
        'text': text,
        'model_id': MODEL_ID,
        'voice_settings': {
            'stability': 0.65,
            'similarity_boost': 0.8,
            'style': 0,
            'speed': 0.9,
        },
    }).encode()
    request = urllib.request.Request(
        f'https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128',
        data=request_body,
        headers={'xi-api-key': api_key, 'Content-Type': 'application/json'},
        method='POST',
    )
    mp3 = target.with_suffix('.mp3')
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                mp3.write_bytes(response.read())
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f'ElevenLabs HTTP {error.code} while generating {target.stem}') from None
            time.sleep(min(30, 3 * 2 ** attempt))
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(mp3),
        '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le', str(target),
    ], check=True)
    mp3.unlink()


def main():
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    if not api_key:
        raise SystemExit('Set ELEVENLABS_API_KEY before generating audio.')
    CLIPS.mkdir(parents=True, exist_ok=True)
    needed = {key: text for _, key, text, _ in CUES}
    for index, (key, text) in enumerate(needed.items(), 1):
        target = CLIPS / f'{key}.wav'
        if target.exists() and target.stat().st_size > 256:
            print(f'[{index:02d}/{len(needed)}] Cached: {key}', flush=True)
            continue
        print(f'[{index:02d}/{len(needed)}] Generating: {key}', flush=True)
        generate(text, target, api_key)
        time.sleep(1)


if __name__ == '__main__':
    main()
