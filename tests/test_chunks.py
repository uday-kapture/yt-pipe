import numpy as np
import pytest

from youtube_transcribe.audio import FRAME_S, MAX_CHUNK_S, plan_chunks


def steady(duration: float) -> np.ndarray:
    return np.ones(int(duration / FRAME_S))


def assert_covers(chunks: list[tuple[float, float]], duration: float) -> None:
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == pytest.approx(duration)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))
    assert all(end - start <= MAX_CHUNK_S for start, end in chunks)


def test_short_episode_is_one_chunk():
    assert plan_chunks(None, 1800.0) == [(0.0, 1800.0)]


def test_cuts_at_the_quietest_frame():
    duration = 2.5 * MAX_CHUNK_S
    rms = steady(duration)
    pause = MAX_CHUNK_S - 30  # inside the first search window
    rms[int(pause / FRAME_S)] = 0.0
    chunks = plan_chunks(rms, duration)
    assert chunks[0][1] == pytest.approx(pause + FRAME_S / 2)
    assert len(chunks) == 3
    assert_covers(chunks, duration)


def test_excluded_region_beats_a_quiet_frame():
    duration = 1.5 * MAX_CHUNK_S
    rms = steady(duration)
    rms[int((MAX_CHUNK_S - 30) / FRAME_S)] = 0.0
    sponsor = (MAX_CHUNK_S - 100, MAX_CHUNK_S - 60)
    chunks = plan_chunks(rms, duration, [sponsor])
    assert chunks[0][1] == pytest.approx(MAX_CHUNK_S - 80)
    assert_covers(chunks, duration)
