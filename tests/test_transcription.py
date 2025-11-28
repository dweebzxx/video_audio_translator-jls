import sys
from unittest.mock import MagicMock

# Mock heavy dependencies BEFORE any imports
sys.modules["faster_whisper"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()

import unittest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from dubber.transcription import TranscriptionManager
from dubber.models import Segment

class TestTranscription(unittest.TestCase):
    @patch("dubber.transcription.WhisperModel")
    @patch("dubber.transcription.Pipeline")
    def test_transcription_manager_init(self, mock_pipeline, mock_whisper):
        # Test default init
        tm = TranscriptionManager(device="cpu", low_mem=False)
        self.assertEqual(tm.model_size, "large-v3")
        self.assertEqual(tm.compute_type, "float32")

        # Test low mem
        tm_low = TranscriptionManager(device="cpu", low_mem=True)
        self.assertEqual(tm_low.model_size, "small")
        self.assertEqual(tm_low.compute_type, "int8")

    @patch("dubber.transcription.WhisperModel")
    def test_transcribe(self, mock_whisper_cls):
        # Mock instance
        mock_instance = MagicMock()
        mock_whisper_cls.return_value = mock_instance

        # Mock result segment
        SegmentMock = MagicMock()
        SegmentMock.start = 0.0
        SegmentMock.end = 2.0
        SegmentMock.text = "Hello world"

        # transcribe returns generator
        mock_instance.transcribe.return_value = ([SegmentMock], MagicMock(language="en", language_probability=0.9))

        tm = TranscriptionManager()
        segments = tm.transcribe(Path("audio.wav"))

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].source_text, "Hello world")
        self.assertEqual(segments[0].start, 0.0)
        self.assertEqual(segments[0].end, 2.0)

    def test_align_speakers(self):
        # We need to mock TranscriptionManager dependencies since we mock sys.modules?
        # Actually importing TranscriptionManager works now because sys.modules has the mocks.
        tm = TranscriptionManager()

        segments = [
            Segment(id=1, start=0.0, end=2.0, source_text="A"),
            Segment(id=2, start=2.0, end=4.0, source_text="B")
        ]

        mock_diarization = MagicMock()
        mock_region = MagicMock()
        mock_diarization.crop.return_value = mock_region
        mock_region.labels.return_value = ["SPEAKER_01"]

        # mock itertracks
        track_seg = MagicMock()
        track_seg.start = 0.0
        track_seg.end = 2.0

        mock_region.itertracks.return_value = [(track_seg, None, "SPEAKER_01")]

        aligned = tm.align_speakers(segments, mock_diarization)

        self.assertEqual(aligned[0].speaker_id, "SPEAKER_01")

if __name__ == "__main__":
    unittest.main()
