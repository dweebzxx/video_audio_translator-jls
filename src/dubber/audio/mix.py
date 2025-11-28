import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from dubber.models import Segment, StemPaths
from dubber.config import DubbingConfig

logger = logging.getLogger(__name__)

class Mixer:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir

    def create_mixed_audio(self, stem_paths: StemPaths, segments: List[Segment], output_path: Path):
        """
        Mixes the instrumental stem with the synthesized voice segments.
        Handles ducking of instrumental when voice is present.
        """
        logger.info(f"Mixing audio into {output_path}...")

        # 1. Create a single continuous voice track from segments.
        # We start with silence matching the full duration of original audio.
        # But ffmpeg filter 'adelay' and 'amix' is cleaner if we just place them.
        # However, calling ffmpeg with 100 inputs is bad.
        # Better strategy: Generate a silence track of full duration.
        # Then mix each segment into it? Or concatenate with silence pads?
        # Concat is safer.
        # We need to fill gaps with silence.

        # Get duration of original
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(stem_paths.original)
        ]
        try:
            total_duration = float(subprocess.check_output(duration_cmd).decode().strip())
        except:
            logger.warning("Could not determine total duration. Using max segment end.")
            total_duration = segments[-1].end if segments else 0

        # Construct a concat list file
        # Format:
        # file 'path'
        # duration X
        # ...

        # Wait, concat demuxer works if files have same format.
        # Our segments are wavs.
        # We need to fill gaps with silence.

        # Generate a silent wav of 1 sec for padding logic?
        # Or use complex filter filter_complex to place audio?
        # Placing audio at specific timestamps with `adelay` is good but limit on number of inputs.

        # "Offline... efficient".
        # Let's try the Concat approach.
        # Sort segments by start.
        # Current time = 0.
        # For each segment:
        #   gap = seg.start - current_time
        #   if gap > 0: generate silence of 'gap' duration.
        #   append segment audio.
        #   current_time = seg.end
        # Final gap to total_duration.

        concat_list_path = self.work_dir / "concat_list.txt"

        current_time = 0.0

        # Helper to generate silence file
        def get_silence(duration, index):
            path = self.work_dir / f"silence_{index}.wav"
            # ffmpeg -f lavfi -i anullsrc=r=44100:cl=stereo -t duration path
            if not path.exists():
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                    "-t", str(duration), "-q:a", "9", str(path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
            return path

        with open(concat_list_path, "w") as f:
            for i, seg in enumerate(segments):
                if not seg.audio_path or not seg.audio_path.exists():
                    logger.warning(f"Segment {seg.id} missing audio. Skipping.")
                    continue

                # Gap
                gap = seg.start - current_time
                if gap > 0.01: # 10ms tolerance
                    silence_path = get_silence(gap, f"gap_{i}")
                    f.write(f"file '{silence_path.resolve()}'\n")

                f.write(f"file '{seg.audio_path.resolve()}'\n")

                # Update current time
                # We should trust the segment duration or the end timestamp?
                # We time-stretched audio to fit [start, end].
                # So the audio duration should be exactly (end - start).
                # But to be safe, we assume it fills to seg.end.
                current_time = seg.end

            # Final gap
            remaining = total_duration - current_time
            if remaining > 0.01:
                silence_path = get_silence(remaining, "end")
                f.write(f"file '{silence_path.resolve()}'\n")

        # Concatenate to create full voice track
        voice_track = self.work_dir / "full_voice_track.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path), "-c", "copy", str(voice_track)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        # 2. Mix with Instrumental (Duck instrumental)
        # sidechaincompress filter:
        # input 0: instrumental
        # input 1: voice
        # output: mixed
        # filter: [0][1]sidechaincompress=threshold=0.1:ratio=5:attack=50:release=200[ducked];[ducked][1]amix=inputs=2:duration=first[out]
        # Or simpler: amix. But ducking is requested.

        # We use a simple volume reduction when voice is active?
        # sidechaincompress is the standard way.
        # Note: sidechaincompress modifies the *first* input based on the *second*.
        # So we pass instrumental as first, voice as second.
        # Then we mix the *compressed instrumental* with the *original voice*.

        cmd = [
            "ffmpeg", "-y",
            "-i", str(stem_paths.instrumental), # 0
            "-i", str(voice_track),       # 1
            "-filter_complex",
            "[0][1]sidechaincompress=threshold=0.05:ratio=4:attack=50:release=300[ducked];[ducked][1]amix=inputs=2:duration=first[out]",
            "-map", "[out]",
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Mixing failed: {e.stderr.decode()}")
            raise

    def remux_video(self, input_video: Path, mixed_audio: Path, output_video: Path):
        """
        Replaces audio in input_video with mixed_audio, saving to output_video.
        """
        logger.info(f"Remuxing video to {output_video}...")

        # -c:v copy : copy video stream
        # -c:a aac : encode audio to aac (good for mp4)
        # -map 0:v:0 : map first video stream from first input
        # -map 1:a:0 : map first audio stream from second input
        # -shortest : truncate to shortest stream (usually video)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-i", str(mixed_audio),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(output_video)
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"Remuxing failed: {e.stderr.decode()}")
            raise

    def generate_subtitles(self, segments: List[Segment], output_path_base: Path):
        """
        Generates SRT subtitles.
        """
        srt_path = output_path_base.with_suffix(".srt")
        logger.info(f"Generating subtitles at {srt_path}...")

        def format_timestamp(seconds: float) -> str:
            # HH:MM:SS,mmm
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            mins = seconds // 60
            hrs = mins // 60
            mins = mins % 60
            seconds = seconds % 60
            return f"{hrs:02}:{mins:02}:{seconds:02},{millis:03}"

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                start_ts = format_timestamp(seg.start)
                end_ts = format_timestamp(seg.end)
                text = seg.translated_text if seg.translated_text else ""

                f.write(f"{i+1}\n")
                f.write(f"{start_ts} --> {end_ts}\n")
                f.write(f"{text}\n\n")
