import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import torch

# fast-whisper
from faster_whisper import WhisperModel
# pyannote.audio
from pyannote.audio import Pipeline

from dubber.models import Segment
from dubber.utils.device import get_compute_type

logger = logging.getLogger(__name__)

class TranscriptionManager:
    def __init__(self, model_size: str = "large-v3", device: str = "cpu", low_mem: bool = False, use_auth_token: Optional[str] = None):
        self.device = device
        self.compute_type = get_compute_type(device, low_mem)
        # Adjust model size for low_mem if requested
        if low_mem and model_size == "large-v3":
            logger.info("Low memory mode enabled: switching Whisper model to 'small'.")
            self.model_size = "small"
        else:
            self.model_size = model_size

        self.low_mem = low_mem
        self.use_auth_token = use_auth_token
        self._whisper_model = None
        self._diarization_pipeline = None

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            logger.info(f"Loading Faster-Whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            # device_index=0 is required for CUDA, but for MPS usually ignored or 0.
            # faster-whisper might not fully support MPS yet directly?
            # Correction: faster-whisper uses CTranslate2. CTranslate2 supports MacOS/ARM64 but "device=mps" support is experimental or limited.
            # Often 'cpu' is used on M1/M2/M3 for CTranslate2 because it's highly optimized with Apple Accelerate.
            # However, prompt says "Use faster-whisper / whisper-ctranslate2 ... on Apple Silicon".
            # If device='mps' fails, we should fallback to cpu.
            # CTranslate2 usually expects 'cpu' or 'cuda'. 'mps' support is recent.
            # We'll try user requested device, but fallback to cpu if it's mps and not supported.

            try:
                # 'mps' is not a valid device string for CTranslate2 3.x yet (it supports 'cpu', 'cuda', 'auto').
                # Actually, check ctranslate2 docs: "device (str) – Device to use (cpu, cuda, auto)."
                # So we must use 'cpu' for Apple Silicon currently with CTranslate2.
                # The user requirement "Use PyTorch 2.x with MPS backend... for Demucs and XTTS" and "Use faster-whisper...".
                # It doesn't explicitly say "Use faster-whisper on MPS". It just says "Use faster-whisper... for efficient ASR on Apple Silicon".
                # Efficient ASR on Apple Silicon with faster-whisper usually means CPU (Accelerate framework).

                asr_device = "cpu" if self.device == "mps" else self.device
                logger.info(f"Using device '{asr_device}' for Whisper (CTranslate2 does not support MPS directly, uses Accelerate on CPU).")

                self._whisper_model = WhisperModel(
                    self.model_size,
                    device=asr_device,
                    compute_type=self.compute_type
                )
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise

        return self._whisper_model

    @property
    def diarization_pipeline(self):
        if self._diarization_pipeline is None:
            logger.info("Loading Pyannote diarization pipeline...")
            # Pyannote requires authentication usually.
            # If token is not provided, we might fail or need a local offline model.
            # Requirement: "Offline use... never rely on remote APIs".
            # But downloading the model once is allowed.
            # If the user has a token, we use it. If not, this might be tricky.
            # We'll assume the environment has it or it's passed.
            try:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.use_auth_token
                )
                if self.device == "mps" or self.device == "cuda":
                     self._diarization_pipeline.to(torch.device(self.device))
            except Exception as e:
                logger.error(f"Failed to load Diarization pipeline. Do you have a valid HuggingFace token? Error: {e}")
                # Fallback or re-raise?
                # "v1 must support multiple characters... Implement automatic speaker diarization".
                # If we can't load it, we can't diarize.
                raise

        return self._diarization_pipeline

    def transcribe(self, audio_path: Path, language: str = None) -> List[Segment]:
        """
        Transcribes the audio and returns a list of Segments.
        Does NOT perform diarization yet (see transcribe_and_diarize).
        """
        model = self.whisper_model

        # language=None means auto-detect
        segments_gen, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        logger.info(f"Detected language '{info.language}' with probability {info.language_probability}")

        segments_list = []
        for i, seg in enumerate(segments_gen):
            # faster-whisper segment: start, end, text, ...
            if seg.end - seg.start <= 0:
                continue
            if not seg.text.strip():
                continue

            segments_list.append(Segment(
                id=i,
                start=seg.start,
                end=seg.end,
                source_text=seg.text.strip(),
                speaker_id=None # To be filled by diarization
            ))

        return segments_list

    def diarize(self, audio_path: Path, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None):
        """
        Runs diarization on the audio file.
        """
        pipeline = self.diarization_pipeline

        # Logic to guide number of speakers
        # pyannote can infer, or we can hint.
        diarization = pipeline(str(audio_path), num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)

        # diarization is an Annotation object
        # We need to map it to our segments.
        return diarization

    def align_speakers(self, segments: List[Segment], diarization) -> List[Segment]:
        """
        Assigns speaker_ids to segments based on overlap with diarization results.
        """
        # Iterate over segments and find the dominant speaker in that time range.
        for seg in segments:
            # Get the most active speaker during [seg.start, seg.end]
            # crop returns a sub-annotation
            region = diarization.crop(dict(uri=diarization.uri, start=seg.start, end=seg.end))
            labels = region.labels()

            if not labels:
                seg.speaker_id = "SPEAKER_00" # Default/Unknown
                continue

            # Find speaker with max duration in this segment
            # region.chart() gives a list of (label, duration) tuples?
            # No, we iterate.
            speaker_durations = {}
            for segment, _, label in region.itertracks(yield_label=True):
                # Calculate overlap
                overlap_start = max(seg.start, segment.start)
                overlap_end = min(seg.end, segment.end)
                duration = max(0, overlap_end - overlap_start)
                speaker_durations[label] = speaker_durations.get(label, 0) + duration

            if speaker_durations:
                dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                seg.speaker_id = dominant_speaker
            else:
                seg.speaker_id = "SPEAKER_00"

        return segments

    def run(self, audio_path: Path, language: str = None, diarize_audio_path: Optional[Path] = None, max_speakers: int = 3) -> List[Segment]:
        """
        Full transcription + diarization pipeline.
        audio_path: The audio to transcribe (usually full audio or vocals).
        diarize_audio_path: The audio to diarize (usually vocals.wav). If None, uses audio_path.
        """
        # 1. Transcribe
        segments = self.transcribe(audio_path, language=language)

        if not segments:
            return []

        # 2. Diarize
        d_path = diarize_audio_path if diarize_audio_path else audio_path
        try:
            diarization = self.diarize(d_path, max_speakers=max_speakers)

            # 3. Align
            segments = self.align_speakers(segments, diarization)
        except Exception as e:
            logger.warning(f"Diarization failed or skipped: {e}. All segments will have default speaker.")
            for seg in segments:
                seg.speaker_id = "SPEAKER_00"

        return segments
