from pathlib import Path
from typing import Optional
from pydantic import BaseModel

class DubbingConfig(BaseModel):
    """
    Configuration for the dubbing process.
    """
    input_video: Path
    output_dir: Path
    source_language: str
    target_language: str

    # Model choices
    translation_model: str = "nllb-200-distilled-600M" # or "m2m100"
    low_mem: bool = False

    # Optional explicitly provided speaker reference
    speaker_wav: Optional[Path] = None

    # Working directory for intermediate files
    work_dir: Path

    # Subtitles
    generate_subtitles: bool = False

    # Speaker Diarization Config
    max_speakers: int = 3
    hf_token: Optional[str] = None

    # Cache Configuration (optional override)
    cache_dir: Optional[Path] = None

    @property
    def output_video_path(self) -> Path:
        """
        Derives the output video path from the input video name and output directory.
        """
        stem = self.input_video.stem
        return self.output_dir / f"{stem}_dubbed_{self.target_language}.mp4"
