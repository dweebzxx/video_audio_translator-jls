import sys
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules["TTS.api"] = MagicMock()

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from dubber.tts.xtts_engine import XTTSEngine
from dubber.models import Segment

class TestTTS(unittest.TestCase):
    def setUp(self):
        self.work_dir = Path("work_mock")

    @patch("dubber.tts.xtts_engine.XTTSEngine._create_default_reference_audio")
    @patch("dubber.tts.xtts_engine.TTS")
    def test_synthesis(self, mock_tts_cls, mock_create_ref):
        engine = XTTSEngine(work_dir=self.work_dir)

        # Mock TTS instance
        mock_instance = MagicMock()
        mock_tts_cls.return_value = mock_instance

        # Segment
        seg = Segment(id=1, start=0.0, end=5.0, source_text="Hi", translated_text="Hola", speaker_id="SPEAKER_00")
        output_path = Path("out.wav")

        # Mock existence of default speaker file or force profile
        engine.speaker_profiles["SPEAKER_00"] = Path("ref.wav")

        # Mock path.exists
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = True # allow check to pass
            engine.synthesize(seg, "es", output_path)

        mock_instance.tts_to_file.assert_called_once()
        kwargs = mock_instance.tts_to_file.call_args[1]
        self.assertEqual(kwargs['text'], "Hola")
        self.assertEqual(kwargs['language'], "es")

    @patch("subprocess.run")
    @patch("subprocess.check_output")
    def test_adjust_duration(self, mock_check, mock_run):
        engine = XTTSEngine(work_dir=self.work_dir)

        # Current duration 2.0, Target 4.0 -> Slow down (speed 0.5)
        mock_check.return_value = b"2.0"

        with patch("shutil.move") as mock_move:
            engine.adjust_duration(Path("audio.wav"), 4.0)

            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            # args is list of strings.
            # ['ffmpeg', '-y', '-i', 'audio.wav', '-filter:a', 'atempo=0.5', '-vn', 'audio.temp.wav']
            self.assertTrue(any("atempo=0.5" in x for x in args))

if __name__ == "__main__":
    unittest.main()
