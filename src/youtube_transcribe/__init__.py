"""Fetch YouTube podcasts, transcribe them with Nemotron 3.5 Transcribe, and upload segments to a private HF dataset."""

from .cli import main

__all__ = ["main"]
