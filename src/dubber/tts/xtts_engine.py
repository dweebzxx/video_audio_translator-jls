import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict
import wave
import math
import struct
import logging

from TTS.api import TTS

from dubber.models import Segment
from dubber.utils.device import get_device_strategy
from dubber.utils.languages import get_xtts_code

logger = logging.getLogger(__name__)

class XTTSEngine:
    def __init__(self, device: str = "cpu", low_mem: bool = False, work_dir: Path = Path(".")):
        self.device = device
        self.low_mem = low_mem
        self.work_dir = work_dir
        self._tts = None

        self.speaker_profiles: Dict[str, Optional[Path]] = {}

        # Ensure a default speaker reference exists
        self.default_speaker_wav = self.work_dir / "default_speaker.wav"
        if not self.default_speaker_wav.exists():
            self._create_default_reference_audio(self.default_speaker_wav)

    def _create_default_reference_audio(self, path: Path):
        """
        Creates a dummy reference audio file (sine wave) for XTTS to use as a fallback.
        XTTS needs *some* reference to clone, even if we want a 'default' sound.
        Ideally we'd download a real voice sample, but for offline/fallback, a synthetic one prevents crash.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        sample_rate = 22050
        duration = 3.0 # 3 seconds
        frequency = 440.0

        logger.info(f"Creating default speaker reference at {path}")

        with wave.open(str(path), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            n_frames = int(sample_rate * duration)
            data = []
            for i in range(n_frames):
                value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                data.append(struct.pack('<h', value))

            wav_file.writeframes(b''.join(data))

    def load_model(self):
        if self._tts is not None:
            return

        logger.info(f"Loading XTTS v2 model on {self.device}...")

        try:
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

            if self.device == "mps":
                self._tts.to("mps")
            elif self.device == "cuda":
                self._tts.to("cuda")

        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise

    def set_speaker_profile(self, speaker_id: str, speaker_wav: Path):
        """
        Registers a speaker profile.
        """
        if speaker_wav and speaker_wav.exists():
            self.speaker_profiles[speaker_id] = speaker_wav
        else:
            logger.warning(f"Speaker wav {speaker_wav} not found for {speaker_id}.")

    def synthesize(self, segment: Segment, target_lang: str, output_path: Path):
        """
        Synthesizes speech for the segment and saves to output_path.
        """
        self.load_model()

        if not segment.translated_text:
            logger.warning(f"No translated text for segment {segment.id}. Skipping.")
            return

        # Determine speaker reference
        speaker_wav = self.speaker_profiles.get(segment.speaker_id)
        if not speaker_wav:
             # Fallback to default
             logger.info(f"No specific speaker profile for {segment.speaker_id}, using default reference.")
             speaker_wav = self.default_speaker_wav

        if not speaker_wav or not speaker_wav.exists():
             # Should not happen given init logic, but safety check
             raise ValueError(f"No speaker reference audio found for speaker {segment.speaker_id} and no default.")

        lang_code = get_xtts_code(target_lang)

        self._tts.tts_to_file(
            text=segment.translated_text,
            file_path=str(output_path),
            speaker_wav=str(speaker_wav),
            language=lang_code,
            split_sentences=True
        )

        if not output_path.exists():
            raise RuntimeError(f"TTS failed to generate file {output_path}")

    def adjust_duration(self, audio_path: Path, target_duration: float):
        """
        Time-stretches the audio file to match target_duration using ffmpeg.
        """
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
            ], stderr=subprocess.STDOUT).decode().strip()
            current_duration = float(probe)
        except Exception:
            current_duration = 0.0

        if current_duration <= 0.1:
            return

        ratio = current_duration / target_duration
        speed = max(0.5, min(2.0, ratio))

        if 0.9 < speed < 1.1:
            return

        logger.info(f"Time-stretching {audio_path.name}: {current_duration:.2f}s -> {target_duration:.2f}s (Speed: {speed:.2f})")

        temp_path = audio_path.with_suffix(".temp.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-filter:a", f"atempo={speed}",
            "-vn", str(temp_path)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        shutil.move(temp_path, audio_path)
