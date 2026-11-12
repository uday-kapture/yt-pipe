"""Audio helpers: normalize to 16 kHz mono FLAC and split long episodes into ASR-sized chunks.

Chunks are only for the transcription server (≤ 1 h per request). Every time stored in
``episode.json`` refers to the full episode; ``asr_chunks[].start`` maps chunk time back.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE = 16000
# Transcribe's server caps a request at 3600 s (scripts/serve.sh); stay well under it.
MAX_CHUNK_S = 3300.0
# Look this far back from the cap for a good cut point.
SEARCH_S = 120.0
FRAME_S = 0.1


def to_16k_mono(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.flac")  # the .flac suffix tells ffmpeg the output format
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-sample_fmt", "s16", str(tmp)],
        check=True,
    )  # fmt: skip
    tmp.replace(dst)


def frame_rms(path: Path, frame_s: float = FRAME_S) -> np.ndarray:
    """RMS energy per ``frame_s`` frame, read in blocks so long episodes never sit in memory whole."""
    info = soundfile.info(str(path))
    frame = int(info.samplerate * frame_s)
    values = []
    for block in soundfile.blocks(str(path), blocksize=frame * 600, dtype="float32", fill_value=0.0):
        values.append(np.sqrt(np.mean(block.reshape(-1, frame) ** 2, axis=1)))
    return np.concatenate(values)


def plan_chunks(
    rms: np.ndarray | None,
    duration: float,
    excluded: list[tuple[float, float]] = (),
    frame_s: float = FRAME_S,
    max_s: float = MAX_CHUNK_S,
    search_s: float = SEARCH_S,
) -> list[tuple[float, float]]:
    """Contiguous ``(start, end)`` chunks covering ``[0, duration]``, each at most ``max_s`` long.

    Each cut lands in the last ``search_s`` before the cap: inside an excluded region (intro,
    sponsor...) if one overlaps that window, otherwise at the quietest frame.
    """
    chunks, t = [], 0.0
    while duration - t > max_s:
        lo, hi = t + max_s - search_s, t + max_s
        overlaps = [(max(s, lo), min(e, hi)) for s, e in excluded if s < hi and e > lo]
        if overlaps:
            s, e = max(overlaps, key=lambda o: o[1] - o[0])
            cut = (s + e) / 2
        else:
            first, last = int(lo / frame_s), int(hi / frame_s)
            cut = (first + int(np.argmin(rms[first:last])) + 0.5) * frame_s
        chunks.append((round(t, 3), round(cut, 3)))
        t = cut
    chunks.append((round(t, 3), round(duration, 3)))
    return chunks


def write_chunks(src: Path, chunks: list[tuple[float, float]], dst_dir: Path) -> list[Path]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    paths = [dst_dir / f"c{i:03d}.flac" for i in range(len(chunks))]
    if len(chunks) == 1:
        paths[0].unlink(missing_ok=True)
        os.link(src, paths[0])  # same bytes, no extra disk
        return paths
    with soundfile.SoundFile(str(src)) as audio:
        for (start, end), path in zip(chunks, paths, strict=True):
            audio.seek(int(start * audio.samplerate))
            data = audio.read(int((end - start) * audio.samplerate), dtype="int16")
            soundfile.write(str(path), data, audio.samplerate, format="FLAC")
    return paths
