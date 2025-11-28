# Dubber: Local Offline Video Dubbing Pipeline

A high-performance, modular video dubbing tool optimized for **Apple Silicon (M1/M2/M3)** using MPS (Metal Performance Shaders) acceleration. It provides a full pipeline from extracting audio, separating vocals, transcribing, translating, synthesizing new speech with voice cloning, and remuxing back into the video.

## Features

- **Offline**: No cloud APIs. All models run locally.
- **Apple Silicon Optimized**: Uses `MPS` for PyTorch models (XTTS, Demucs) and optimized CPU execution for Faster-Whisper.
- **Modular**: Clean architecture with separation of concerns.
- **Multi-Speaker**: Automatic diarization (up to 3 speakers) and distinct voice synthesis.
- **Voice Cloning**: Optional reference audio for XTTS v2, or fallback to internal defaults.
- **Quality**: Background music/SFX preservation via Demucs source separation and sidechain ducking.

## Requirements

- **macOS** (Apple Silicon recommended)
- **Python 3.11** (via Homebrew)
- **FFmpeg** (via Homebrew)

## Installation

1. **Install System Dependencies**
   ```bash
   brew install python@3.11 ffmpeg
   ```

2. **Clone and Install**
   ```bash
   git clone https://github.com/your-repo/dubber.git
   cd dubber
   pip install -e .
   ```

   *Note*: Some dependencies like `TTS` or `pyannote.audio` might require specific setup. Ensure you are in a clean virtual environment.

3. **Download Models**
   Before the first run, or to prepare for offline usage:
   ```bash
   dubber download-models
   ```
   This will fetch models for Demucs, Faster-Whisper, NLLB/M2M100, and XTTS.

## Usage

### Basic Run
Dub a video from English to Spanish:
```bash
dubber run video.mp4 --source-lang en --target-lang es
```

### Advanced Options
```bash
dubber run video.mp4 \
  --source-lang en \
  --target-lang fr \
  --model nllb-200-distilled-600M \
  --low-mem \
  --subtitles \
  --output-dir ./dubbed_videos \
  --work-dir ./temp_work
```

- `--low-mem`: Use smaller models (Whisper Small, quantization) for 8GB/16GB machines.
- `--subtitles`: Generate SRT subtitles alongside the video.
- `--speaker-wav <path>`: Provide a reference audio file for voice cloning.
- `--work-dir <path>`: Specify a directory for intermediate files (useful for external drives).

## Pipeline Architecture

1. **Audio Extraction**: FFmpeg extracts WAV from video.
2. **Separation**: Demucs separates `vocals` and `instrumental` (music/sfx).
3. **Diarization & Transcription**: Faster-Whisper + Pyannote identify speakers and text.
4. **Translation**: NLLB-200 translates text to target language.
5. **Synthesis**: Coqui XTTS v2 generates speech, maintaining speaker identity if configured.
6. **Mixing**: FFmpeg mixes `instrumental` and synthesized speech with ducking.
7. **Remuxing**: Final audio is replaced in the original video container.

## Troubleshooting

- **Memory Issues**: Use `--low-mem`.
- **MPS Errors**: The tool tries to use MPS. If it fails, it falls back to CPU. Check PyTorch installation.
- **Demucs/TTS Downloads**: If downloads fail, check internet connection. You can manually place models in `~/.cache/...`.

## License
MIT
