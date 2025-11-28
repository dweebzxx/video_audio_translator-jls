from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator

class StemPaths(BaseModel):
    """
    Paths to the separated audio stems.
    """
    original: Path
    vocals: Path
    instrumental: Path
    other: Optional[Dict[str, Path]] = Field(default_factory=dict)

    @field_validator("original", "vocals", "instrumental")
    def path_must_exist(cls, v: Path) -> Path:
        return v

class Segment(BaseModel):
    """
    A single segment of speech in the video.
    """
    id: int
    start: float
    end: float
    source_text: str = ""
    translated_text: Optional[str] = None
    speaker_id: Optional[str] = None

    # Path to the synthesized audio for this segment
    audio_path: Optional[Path] = None

    @model_validator(mode='after')
    def check_timestamps(self) -> 'Segment':
        if self.end <= self.start:
            raise ValueError(f"Segment end ({self.end}) must be greater than start ({self.start})")
        return self
