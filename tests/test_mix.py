import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from dubber.audio.mix import Mixer
from dubber.models import Segment, StemPaths

class TestMix(unittest.TestCase):
    @patch("subprocess.run")
    @patch("subprocess.check_output")
    def test_create_mixed_audio(self, mock_check, mock_run):
        mixer = Mixer(Path("work"))

        stems = StemPaths(original=Path("orig.wav"), vocals=Path("v.wav"), instrumental=Path("i.wav"))
        segments = [
            Segment(id=1, start=0.0, end=1.0, audio_path=Path("s1.wav"))
        ]

        # Mock total duration
        mock_check.return_value = b"10.0"

        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            # Mock exists for segments
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                mixer.create_mixed_audio(stems, segments, Path("mixed.wav"))

        # Expect multiple calls to subprocess.run:
        # 1. silence gen (maybe)
        # 2. concat
        # 3. mix
        self.assertGreaterEqual(mock_run.call_count, 2)

        # Check mixing command for sidechain
        # The mix command is the LAST call to subprocess.run usually?
        # Let's inspect all calls
        calls = mock_run.call_args_list
        found = False
        for call in calls:
            args = call[0][0]
            # args is list of strings
            # Look for filter_complex
            if "-filter_complex" in args:
                idx = args.index("-filter_complex")
                filter_str = args[idx+1]
                if "sidechaincompress" in filter_str:
                    found = True
                    break

        self.assertTrue(found, "sidechaincompress filter not found in ffmpeg calls")

    def test_generate_subtitles(self):
        mixer = Mixer(Path("work"))
        segments = [
            Segment(id=1, start=0.5, end=2.5, translated_text="Hello")
        ]

        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            mixer.generate_subtitles(segments, Path("vid.mp4"))

            mock_file().write.assert_any_call("00:00:00,500 --> 00:00:02,500\n")
            mock_file().write.assert_any_call("Hello\n\n")

if __name__ == "__main__":
    unittest.main()
