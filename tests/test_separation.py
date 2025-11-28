import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from dubber.audio.separation import separate_audio, extract_audio

class TestSeparation(unittest.TestCase):
    @patch("subprocess.run")
    def test_extract_audio(self, mock_run):
        work_dir = Path("work")
        video = Path("video.mp4")

        out = extract_audio(video, work_dir)

        self.assertEqual(out, work_dir / "video.wav")
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("ffmpeg", args)
        self.assertIn("-vn", args)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_separate_audio(self, mock_exists, mock_run):
        # Mock existence checks to pass
        mock_exists.return_value = True

        input_audio = Path("audio.wav")
        output_dir = Path("out")

        # separate_audio checks if output files exist at the end.
        # We need to ensure logic flow works.

        stems = separate_audio(input_audio, output_dir, device="cpu", low_mem=True)

        self.assertEqual(stems.original, input_audio)
        self.assertTrue(str(stems.vocals).endswith("vocals.wav"))

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertIn("demucs", args)
        self.assertIn("--two-stems", args) # Verify requirement
        self.assertIn("vocals", args)

if __name__ == "__main__":
    unittest.main()
