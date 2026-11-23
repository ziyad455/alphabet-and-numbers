# Numbers 1–20 narration kit

The published Slide 8 video is `../chapter-1/slide-8/assets/numbers-1-20-elevenlabs-voice.mp4`. It retains the video stream from `../chapter-1/slide-8/assets/numbers.mp4` and replaces its original audio stream with one synchronized narration track. The original MP4 remains in the assets directory as the source and backup.

The final narration uses **ElevenLabs Alice** (`Xb7hH8MSUJpSbSDYk0k2`), a British English educator voice, with `eleven_multilingual_v2`. The generator makes 23 unique clips (intro, numbers 1–20, counting prompt, outro) and places them at 45 points in the 152.183-second video. The repeated fourteen and seventeen cues support the extended spelling screens; the final count runs from one to twenty.

## Rebuild

From this directory, with Python 3.10+, NumPy, SoundFile, and FFmpeg installed:

1. Set `ELEVENLABS_API_KEY` in the shell environment. Do not put the key in source files or commit it.
2. Run `python generate_elevenlabs_clips.py`. Completed WAV clips are cached under `generated-voice/elevenlabs/clips/`, so reruns resume without regenerating them.
3. Run `python generate_gemini_video.py --use-existing-clips --work-dir generated-voice/elevenlabs --output ../chapter-1/slide-8/assets/numbers-1-20-elevenlabs-voice.mp4`.

The assembly script also retains optional Gemini TTS generation support. Its shared `CUES` timeline was corrected after checking frames from the actual video. The Gemini API hit a quota limit during this production run; **the published video contains only ElevenLabs narration**. All generated clips, intermediate WAV files, and duplicate source videos in this kit are ignored by Git.

## Verification performed

- Checked frames around each number entrance and the final count.
- Transcribed the assembled voice track: numbers 1–20 appear in order, followed by the complete 1–20 count and the closing encouragement.
- Compared stream hashes: the final video stream matches the source exactly; the final audio stream differs. The final MP4 contains one audio stream.
- Direct human listening is still recommended before classroom use.
