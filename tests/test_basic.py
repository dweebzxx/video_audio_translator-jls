import unittest
from pathlib import Path
from dubber.config import DubbingConfig
from dubber.models import Segment
from dubber.utils.languages import normalize_lang_code, get_xtts_code

class TestConfig(unittest.TestCase):
    def test_segment_validation(self):
        # Valid
        s = Segment(id=1, start=0.0, end=1.0, source_text="test")
        self.assertEqual(s.end, 1.0)

        # Invalid timestamps
        with self.assertRaises(ValueError):
            Segment(id=2, start=1.0, end=0.5, source_text="fail")

    def test_dubbing_config(self):
        c = DubbingConfig(
            input_video=Path("video.mp4"),
            output_dir=Path("out"),
            source_language="en",
            target_language="es",
            work_dir=Path("work")
        )
        self.assertEqual(c.output_video_path.name, "video_dubbed_es.mp4")

class TestLanguages(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_lang_code("EN"), "en")
        self.assertEqual(normalize_lang_code(" es "), "es")

    def test_xtts_code(self):
        self.assertEqual(get_xtts_code("en"), "en")
        with self.assertRaises(ValueError):
            get_xtts_code("xx") # Invalid

if __name__ == "__main__":
    unittest.main()
