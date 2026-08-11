# ABOUTME: Chooses the torch device every AutoMorph inference module runs on.
# ABOUTME: AUTOMORPH_DEVICE pins the choice; otherwise CUDA, then MPS, then CPU.

import logging
import os

import torch

SUPPORTED_TYPES = ("cpu", "cuda", "mps")


def select_device(local_rank=0):
    """Return the torch device to run inference on.

    AUTOMORPH_DEVICE pins the device explicitly, e.g. "cpu", "cuda:0" or "mps". Without it the
    best available accelerator is used. ``local_rank`` names the CUDA device when auto-detection
    picks CUDA; it is ignored otherwise.

    :raises RuntimeError: if AUTOMORPH_DEVICE names a device type torch cannot provide here.
    """
    requested = os.getenv("AUTOMORPH_DEVICE")
    if requested:
        device = _parse(requested)
        logging.info(f"AUTOMORPH_DEVICE is set. Using {device}...")
        return device

    if torch.cuda.is_available():
        logging.info("CUDA is available. Using CUDA...")
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        logging.info("MPS is available. Using MPS...")
        return torch.device("mps")
    logging.info("Neither CUDA nor MPS is available. Using CPU...")
    return torch.device("cpu")


def _parse(requested):
    """Turn an AUTOMORPH_DEVICE value into a device torch can actually allocate on."""
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise RuntimeError(f"AUTOMORPH_DEVICE={requested!r} is not a torch device: {error}") from error

    if device.type not in SUPPORTED_TYPES:
        raise RuntimeError(f"AUTOMORPH_DEVICE={requested!r} names an unsupported device type")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"AUTOMORPH_DEVICE={requested!r} but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"AUTOMORPH_DEVICE={requested!r} but MPS is not available")
    return device
