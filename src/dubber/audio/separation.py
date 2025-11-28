import shutil
import subprocess
from pathlib import Path
from typing import Optional
import logging
import torch

from dubber.models import StemPaths

logger = logging.getLogger(__name__)

def separate_audio(
    input_audio: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    device: str = "cpu",
    low_mem: bool = False
) -> StemPaths:
    """
    Separates audio into vocals and instrumental using Demucs.
    Returns the paths to the separated stems.
    """
    logger.info(f"Starting audio separation with Demucs (model={model_name}, device={device})...")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "demucs",
        "--name", model_name,
        "--out", str(output_dir),
        "--two-stems", "vocals", # Force 2 stems: vocals and no_vocals
        str(input_audio)
    ]

    if device == "mps":
        cmd.extend(["-d", "mps"])
    else:
        cmd.extend(["-d", "cpu"])

    if low_mem:
        # Reduce shifts to 0 to save time/memory, default is usually 1 or 0 depending on model
        cmd.extend(["--shifts", "0"])
        # potentially limit jobs
        cmd.extend(["--jobs", "1"])

    logger.debug(f"Running command: {' '.join(cmd)}")
    try:
        # We capture stdout/stderr to avoid clutter unless error
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs separation failed: {e.stderr}")
        raise RuntimeError(f"Demucs failed with exit code {e.returncode}") from e

    # Demucs output structure: {output_dir}/{model_name}/{track_name}/...
    track_name = input_audio.stem
    expected_model_dir = output_dir / model_name / track_name

    vocals_path = expected_model_dir / "vocals.wav"
    instrumental_path = expected_model_dir / "no_vocals.wav"

    if not vocals_path.exists() or not instrumental_path.exists():
        raise RuntimeError(f"Expected Demucs output files not found in {expected_model_dir}")

    return StemPaths(
        original=input_audio,
        vocals=vocals_path,
        instrumental=instrumental_path
    )

def extract_audio(video_path: Path, work_dir: Path) -> Path:
    """
    Extracts audio from video file to WAV format.
    """
    logger.info(f"Extracting audio from {video_path}...")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_wav = work_dir / f"{video_path.stem}.wav"

    if output_wav.exists():
        logger.info(f"Audio file {output_wav} already exists. Skipping extraction.")
        return output_wav

    cmd = [
        "ffmpeg",
        "-y", # Overwrite
        "-i", str(video_path),
        "-vn", # No video
        "-acodec", "pcm_s16le", # WAV standard
        "-ar", "44100", # 44.1kHz
        "-ac", "2", # Stereo
        str(output_wav)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg extraction failed: {e.stderr.decode()}")
        raise RuntimeError("Audio extraction failed") from e

    return output_wav
