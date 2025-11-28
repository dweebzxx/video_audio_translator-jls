import torch
import logging

logger = logging.getLogger(__name__)

def get_device_strategy() -> str:
    """
    Determines the best available device for execution (MPS > CPU).
    Returns the device string (e.g., 'mps' or 'cpu').
    """
    if torch.backends.mps.is_available():
        logger.info("Apple Silicon MPS backend detected and available.")
        return "mps"
    elif torch.cuda.is_available():
        # Strictly avoiding CUDA as per requirements, but good to know if it exists in environment.
        # We will NOT return cuda, we will warn and use cpu to strictly follow "MPS or CPU only"
        logger.warning("CUDA detected but strictly creating pipeline for Apple Silicon (MPS/CPU). Using CPU.")
        return "cpu"
    else:
        logger.info("MPS not available. Falling back to CPU.")
        return "cpu"

def get_compute_type(device: str, low_mem: bool = False) -> str:
    """
    Returns the compute type for CTranslate2/Whisper based on device and memory mode.
    """
    if device == "mps":
        # CTranslate2 on MPS usually uses float16 or float32.
        # Int8 might not be supported on MPS in all versions.
        return "float16" if not low_mem else "int8"
    else:
        return "int8" if low_mem else "float32"
