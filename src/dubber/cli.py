import argparse
import sys
import logging
from pathlib import Path
import shutil

from dubber.config import DubbingConfig
from dubber.pipeline import Pipeline
from dubber.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Dubber: Local Offline Video Dubbing Pipeline")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the dubbing pipeline")
    run_parser.add_argument("input_video", type=Path, help="Path to input video file")
    run_parser.add_argument("--source-lang", "-s", required=True, help="Source language code (ISO 639-1)")
    run_parser.add_argument("--target-lang", "-t", required=True, help="Target language code (ISO 639-1)")
    run_parser.add_argument("--output-dir", "-o", type=Path, default=Path("output"), help="Output directory")
    run_parser.add_argument("--work-dir", "-w", type=Path, default=Path("work"), help="Working directory for temp files")
    run_parser.add_argument("--speaker-wav", type=Path, help="Optional reference audio for voice cloning")
    run_parser.add_argument("--model", default="nllb-200-distilled-600M", help="Translation model name")
    run_parser.add_argument("--low-mem", action="store_true", help="Enable low memory mode")
    run_parser.add_argument("--subtitles", action="store_true", help="Generate subtitles")
    run_parser.add_argument("--max-speakers", type=int, default=3, help="Maximum number of speakers to detect")
    run_parser.add_argument("--hf-token", help="HuggingFace token for pyannote diarization")

    # Download models command
    download_parser = subparsers.add_parser("download-models", help="Download all required models")
    download_parser.add_argument("--low-mem", action="store_true", help="Download smaller models where applicable")

    args = parser.parse_args()

    setup_logging("INFO")

    if args.command == "run":
        if not args.input_video.exists():
            logger.error(f"Input video not found: {args.input_video}")
            sys.exit(1)

        config = DubbingConfig(
            input_video=args.input_video.resolve(),
            output_dir=args.output_dir.resolve(),
            source_language=args.source_lang,
            target_language=args.target_lang,
            translation_model=args.model,
            low_mem=args.low_mem,
            speaker_wav=args.speaker_wav.resolve() if args.speaker_wav else None,
            work_dir=args.work_dir.resolve(),
            generate_subtitles=args.subtitles,
            max_speakers=args.max_speakers,
            hf_token=args.hf_token
        )

        try:
            pipeline = Pipeline(config)
            pipeline.run()
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "download-models":
        logger.info("Downloading models...")
        # Trigger downloads by instantiating or calling load methods
        # This is a bit of a hack, but effective.

        from dubber.translation import Translator
        from dubber.transcription import TranscriptionManager
        from dubber.tts.xtts_engine import XTTSEngine
        # Demucs is cmd line, but we can trigger it or assume it downloads on first run.
        # Actually demucs has a python api to download.

        try:
            # 1. Translation
            logger.info("Checking Translation model...")
            t = Translator(model_name="nllb-200-distilled-600M", low_mem=args.low_mem)
            t.load_model()

            # 2. Transcription
            logger.info("Checking Transcription model...")
            tm = TranscriptionManager(low_mem=args.low_mem)
            _ = tm.whisper_model
            # Diarization
            logger.info("Checking Diarization model...")
            try:
                _ = tm.diarization_pipeline
            except Exception as e:
                logger.warning(f"Diarization model download check failed (auth might be needed later): {e}")

            # 3. TTS
            logger.info("Checking TTS model...")
            # We construct but don't synthesis
            tts = XTTSEngine(low_mem=args.low_mem)
            tts.load_model()

            # 4. Demucs
            logger.info("Checking Demucs...")
            # We can run a dummy command to force download
            # subprocess.run(["demucs", "--help"], ...) doesn't download.
            # Running on a dummy file does.
            # But we don't have a dummy file easily.
            # Demucs usually downloads to cache on first real run.
            # We can try importing demucs.pretrained
            # from demucs.pretrained import get_model; get_model('htdemucs')
            try:
                from demucs.pretrained import get_model
                get_model('htdemucs')
                logger.info("Demucs model verified.")
            except ImportError:
                logger.warning("Could not import demucs python API to preload model. It will download on first run.")
            except Exception as e:
                logger.warning(f"Demucs download check failed: {e}")

            logger.info("All models checked/downloaded.")

        except Exception as e:
            logger.error(f"Download failed: {e}")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
