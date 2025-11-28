import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["faster_whisper"] = MagicMock()
sys.modules["pyannote.audio"] = MagicMock()
sys.modules["TTS.api"] = MagicMock()

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from dubber.pipeline import Pipeline
from dubber.config import DubbingConfig
from dubber.models import Segment, StemPaths

class TestIntegration(unittest.TestCase):
    """
    Simulates the full pipeline by mocking the heavy ML components
    but executing the orchestration logic.
    """

    @patch("dubber.pipeline.extract_audio")
    @patch("dubber.pipeline.separate_audio")
    @patch("dubber.pipeline.TranscriptionManager")
    @patch("dubber.pipeline.Translator")
    @patch("dubber.pipeline.XTTSEngine")
    @patch("dubber.pipeline.Mixer")
    def test_pipeline_flow(self, MockMixer, MockTTS, MockTranslator, MockTranscriber, mock_separate, mock_extract):
        # Setup Config
        config = DubbingConfig(
            input_video=Path("test.mp4"),
            output_dir=Path("out"),
            source_language="en",
            target_language="fr",
            work_dir=Path("work")
        )

        # Mock Returns
        mock_extract.return_value = Path("work/test.wav")

        mock_separate.return_value = StemPaths(
            original=Path("work/test.wav"),
            vocals=Path("work/vocals.wav"),
            instrumental=Path("work/instr.wav")
        )

        # Mock Transcriber
        transcriber_instance = MockTranscriber.return_value
        transcriber_instance.run.return_value = [
            Segment(id=0, start=0.0, end=1.0, source_text="Hello", speaker_id="SPEAKER_00")
        ]

        # Mock Translator
        translator_instance = MockTranslator.return_value
        translator_instance.translate_text.return_value = "Bonjour"

        # Mock TTS
        tts_instance = MockTTS.return_value

        # Run Pipeline
        pipeline = Pipeline(config)
        pipeline.run()

        # Assertions
        mock_extract.assert_called_once()
        mock_separate.assert_called_once()
        transcriber_instance.run.assert_called_once()
        translator_instance.translate_text.assert_called_once_with("Hello", src_lang="en", tgt_lang="fr")

        # Check TTS call
        tts_instance.synthesize.assert_called_once()
        args = tts_instance.synthesize.call_args[0]
        self.assertEqual(args[0].translated_text, "Bonjour")
        self.assertEqual(args[1], "fr")

        # Check Mixer
        mixer_instance = MockMixer.return_value
        mixer_instance.create_mixed_audio.assert_called_once()
        mixer_instance.remux_video.assert_called_once()

if __name__ == "__main__":
    unittest.main()
