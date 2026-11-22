"""Build the ElevenLabs soundtrack and mux it without re-encoding video.

Run: python build_audio.py ../assets/introducing-yourself.mp4
Requires requests and FFmpeg (system FFmpeg or imageio-ffmpeg).
Reads ELEVENLABS_API_KEY, or the existing local .env `api` field.
"""
import json
import os
import re
import subprocess
import sys
import wave
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CUES = json.loads((HERE / 'cues.json').read_text())
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = 'ffmpeg'


def key():
    value = os.environ.get('ELEVENLABS_API_KEY')
    if value:
        return value
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                name, value = line.split('=', 1)
                if name.strip() in ('ELEVENLABS_API_KEY', 'api'):
                    return value.strip().strip('"\'')
    raise RuntimeError('ELEVENLABS_API_KEY is unavailable')


def run(args, log=None):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log is not None:
        log.write(result.stderr + '\n')
    if result.returncode:
        raise RuntimeError(f'FFmpeg failed: {result.stderr[-1500:]}')
    return result


def duration(path):
    with wave.open(str(path)) as source:
        return source.getnframes() / source.getframerate()


def make_clip(cue, api_key, log):
    clip = CACHE / f"{cue['id']}.wav"
    if clip.exists():
        return clip
    mp3 = CACHE / f"{cue['id']}.mp3"
    if not mp3.exists():
        voice = CUES['voices'][cue['voice']]['id']
        response = requests.post(
            f'https://api.elevenlabs.io/v1/text-to-speech/{voice}',
            params={'output_format': 'mp3_44100_128'},
            headers={'xi-api-key': api_key, 'Content-Type': 'application/json'},
            json={'text': cue['text'], 'model_id': 'eleven_multilingual_v2',
                  'voice_settings': {'stability': 0.58, 'similarity_boost': 0.78, 'speed': 1.05}},
            timeout=90,
        )
        if not response.ok:
            raise RuntimeError(f"ElevenLabs request for {cue['id']} failed: HTTP {response.status_code}, {response.text[:220]}")
        mp3.write_bytes(response.content)
    trimmed = CACHE / f"{cue['id']}-trim.wav"
    run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(mp3), '-af',
         'silenceremove=start_periods=1:start_threshold=-45dB:start_duration=0.02,'
         'areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_duration=0.02,areverse',
         '-ar', '48000', '-ac', '1', str(trimmed)], log)
    length = duration(trimmed)
    factor = length / cue['max_dur']
    if factor > 1.25:
        raise RuntimeError(f"!! {cue['id']} lasts {length:.2f}s, exceeds {cue['max_dur']:.2f}s by too much; adjust voice or narrator wording")
    filter_audio = f'atempo={factor * 1.015:.5f},' if factor > 1 else ''
    filter_audio += 'loudnorm=I=-20:TP=-3:LRA=11'
    run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(trimmed), '-af', filter_audio,
         '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le', str(clip)], log)
    final = duration(clip)
    if final > cue['max_dur'] + .005:
        raise RuntimeError(f"!! {cue['id']} still exceeds its window: {final:.3f}s > {cue['max_dur']:.3f}s")
    log.write(f"{cue['id']} {cue['voice']} {final:.3f}/{cue['max_dur']:.3f}s\n")
    print(f"{cue['id']} {final:.2f}s", flush=True)
    return clip


def main():
    video = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / 'assets' / 'introducing-yourself.mp4'
    if not video.is_file():
        raise RuntimeError(f'Video missing: {video}')
    api_key = key()
    output = HERE.parent / 'assets' / 'introducing-yourself-with-audio.mp4'
    mix = HERE / 'mix.wav'
    with (HERE / 'build.log').open('w') as log:
        source_info = run([FFMPEG, '-hide_banner', '-i', str(video), '-f', 'null', '-'], log).stderr
        match = re.search(r'Duration: (\d+):(\d+):([\d.]+)', source_info)
        if not match:
            raise RuntimeError('Could not read source video duration')
        seconds = int(match[1])*3600 + int(match[2])*60 + float(match[3])
        if abs(seconds - CUES['duration']) > .05:
            raise RuntimeError(f'Video duration {seconds} differs from cue sheet {CUES["duration"]}')
        clips = {cue['id']: make_clip(cue, api_key, log) for cue in CUES['cues']}
        placements = []
        for cue in CUES['cues']:
            placements.append((cue['start'], clips[cue['id']]))
            if cue['again'] is not None:
                placements.append((cue['again'], clips[cue['id']]))
        args = [FFMPEG, '-y', '-loglevel', 'error']
        filters = []
        for i, (start, clip) in enumerate(placements):
            args += ['-i', str(clip)]
            delay = round(start * 1000)
            filters.append(f'[{i}:a]adelay={delay}:all=1[a{i}]')
        inputs = ''.join(f'[a{i}]' for i in range(len(placements)))
        filters.append(f'{inputs}amix=inputs={len(placements)}:normalize=0,apad,atrim=duration={CUES["duration"]},loudnorm=I=-16:TP=-1.5:LRA=11,aresample=48000[mix]')
        args += ['-filter_complex', ';'.join(filters), '-map', '[mix]', '-ac', '2', '-ar', '48000', '-c:a', 'pcm_s16le', str(mix)]
        run(args, log)
        run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(video), '-i', str(mix),
             '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-t', str(CUES['duration']), '-movflags', '+faststart', str(output)], log)
    print('Built', output)


if __name__ == '__main__':
    main()
