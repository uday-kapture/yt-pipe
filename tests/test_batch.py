import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile

from youtube_transcribe.batch import (
    CHUNK_FEATURES,
    MAX_SEGMENT_S,
    SEGMENT_FEATURES,
    SEGMENT_PAD_S,
    _chunk_rows,
    _segment_rows,
    last_batch_on_hub,
    next_batch,
    speaker_turns,
    turn_segments,
    write_parquet,
)

RATE = 16000
EPISODE = {
    "episode_id": "yt_abc",
    "youtube": {"id": "abc", "channel": "Pod", "channel_id": "UC1", "title": "Ep 1", "upload_date": "20260101"},
    "source": {"name": "pod", "lang_hint": "hi-en"},
}


def row(i: int, hours: float) -> dict:
    return {"episode_id": f"yt_{i}", "duration": hours * 3600}


def ids(rows: list[dict]) -> list[str]:
    return [r["episode_id"] for r in rows]


def test_next_batch_skips_done_and_stops_at_budget():
    catalog = [row(0, 2), row(1, 3), row(2, 4), row(3, 5), row(4, 1)]
    assert ids(next_batch(catalog, {"yt_1"}, batch_hours=7)) == ["yt_0", "yt_2"]


def test_last_batch_on_hub_ignores_other_files():
    files = ["README.md", "meta/batch-00002.jsonl", "data/chunks/batch-00009.parquet", "meta/batch-00012.jsonl"]
    assert last_batch_on_hub(files) == 12
    assert last_batch_on_hub(["README.md"]) == 0


def test_next_batch_takes_one_episode_over_budget():
    assert ids(next_batch([row(0, 12)], set(), batch_hours=10)) == ["yt_0"]
    assert next_batch([row(0, 1)], {"yt_0"}, batch_hours=10) == []


def ipu(id_: int, speaker: int, onset: str, words: list[tuple], trailing: str = "") -> dict:
    """An IPU as nemo's annotate step writes it (only the fields segments use)."""
    return {"id": id_, "speaker": speaker, "onset": onset, "start": words[0][1], "end": words[-1][2],
            "words": [list(w) for w in words], "word_groups": [w[0] for w in words],
            "trailing_tags": trailing}  # fmt: skip


# speaker 0 pauses mid-sentence, speaker 1 says "haan" in the pause, speaker 0 goes on, then speaker 1 takes over
IPUS = [
    ipu(0, 0, "start", [("so", 0.0, 0.3), ("the", 0.4, 0.6), ("plan", 0.7, 1.5)]),
    ipu(1, 1, "backchannel", [("haan", 1.8, 2.1)]),
    ipu(2, 0, "continue", [("is", 2.4, 2.7), ("simple", 2.8, 3.4)], trailing="<ev:laughter>"),
    ipu(3, 1, "switch", [("okay", 4.0, 4.4), ("great", 4.5, 5.0)]),
]


@pytest.fixture
def nemo_output(tmp_path):
    """Fake nemo outputs: chunk c001 (10 s, starting 3291.4 s into the episode) is ok, c002 failed alignment."""
    chunk_path = tmp_path / "c001.flac"
    noise = np.random.default_rng(0).standard_normal(10 * RATE) * 1000
    soundfile.write(str(chunk_path), noise.astype("int16"), RATE)
    asr = tmp_path / "asr"
    (asr / "conversations").mkdir(parents=True)
    conversation = {"id": "yt_abc__c001", "audio": str(chunk_path), "status": "ok", "duration": 10.0,
                    "transcript": "<spk:0> so the plan", "transcript_tagged": "<spk:0> so the plan",
                    "speakers": {"0": {}, "1": {}}, "truncated": False, "pass_agreement": 0.9,
                    "ipus": IPUS}  # fmt: skip
    (asr / "conversations" / "yt_abc__c001.json").write_text(json.dumps(conversation))
    failed = {"id": "yt_abc__c002", "audio": str(chunk_path), "status": "alignment_failed:no_words"}
    (asr / "conversations" / "yt_abc__c002.json").write_text(json.dumps(failed))
    chunks = [
        {"id": "yt_abc__c001", "start": 3291.4, "end": 3301.4},
        {"id": "yt_abc__c002", "start": 3301.4, "end": 4000},
    ]
    return asr, {"yt_abc": EPISODE | {"asr_chunks": chunks}}, chunk_path


def test_chunk_rows_keep_the_chunk_file_and_skip_failed_chunks(nemo_output):
    asr, episodes, chunk_path = nemo_output
    [chunk] = _chunk_rows(asr, episodes)
    assert chunk["audio"]["bytes"] == chunk_path.read_bytes()  # uploaded as-is, no re-encode
    assert (chunk["chunk_id"], chunk["start"], chunk["duration"]) == ("yt_abc__c001", 3291.4, 10.0)
    assert "audio" not in json.loads(chunk["annotation"])  # no local paths on HF
    assert set(CHUNK_FEATURES) == set(chunk)


def test_segments_are_speaker_turns(nemo_output):
    asr, episodes, _ = nemo_output
    rows = _segment_rows(asr, episodes)
    assert [(r["speaker"], r["text"], r["onset"], r["ipus"]) for r in rows] == [
        (0, "so the plan is simple", "start", [0, 2]),  # the pause and the listener's "haan" don't split the turn
        (1, "haan", "backchannel", [1]),
        (1, "okay great", "switch", [3]),
    ]
    turn, backchannel, _ = rows
    assert turn["text_tagged"] == "so the plan is simple <ev:laughter>"
    assert backchannel["text_tagged"] == "<int:backchannel> haan"
    assert (turn["overlap"], backchannel["overlap"]) == (True, False)  # the "haan" is audible inside the turn
    assert (turn["start"], turn["end"]) == (3291.4, 3294.9)  # episode time; padding clamped at the chunk start
    assert (turn["video_id"], turn["lang_hint"], turn["pass_agreement"]) == ("abc", "hi-en", 0.9)
    clip, clip_rate = soundfile.read(io.BytesIO(turn["audio"]["bytes"]))
    assert (len(clip), clip_rate) == (int(3.5 * RATE), RATE)
    assert set(SEGMENT_FEATURES) == set(turn)


def test_continue_extends_the_last_floor_turn_not_a_backchannel():
    ipus = [
        ipu(0, 1, "switch", [("so", 0.0, 3.0)]),
        ipu(1, 0, "failed_interruption", [("wait", 2.0, 2.5)]),
        ipu(2, 1, "backchannel", [("hmm", 4.0, 4.3)]),
        ipu(3, 1, "continue", [("anyway", 5.0, 6.0)]),
    ]
    assert [[i["id"] for i in turn] for turn in speaker_turns(ipus)] == [[0, 3], [1], [2]]


def test_long_turns_split_at_word_boundaries():
    words = [(f"w{i}", float(i), i + 0.8) for i in range(30)]  # 30 s of continuous speech
    turn = [ipu(0, 0, "switch", words)]
    pieces = turn_segments(turn, turn, chunk_duration=100.0)
    assert [p["onset"] for p in pieces] == ["switch", "continue"]
    assert all(p["duration"] <= MAX_SEGMENT_S + 2 * SEGMENT_PAD_S for p in pieces)
    assert " ".join(p["text"] for p in pieces) == " ".join(w[0] for w in words)


@pytest.mark.parametrize(("build", "features"), [(_chunk_rows, CHUNK_FEATURES), (_segment_rows, SEGMENT_FEATURES)])
def test_parquet_has_a_playable_audio_column(nemo_output, tmp_path, build, features):
    asr, episodes, _ = nemo_output
    rows = build(asr, episodes)
    parquet = tmp_path / "subset.parquet"
    write_parquet(rows, features, parquet)

    schema = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"]
    assert schema["audio"]["_type"] == "Audio"  # what the HF viewer needs to play it
    stored = pq.read_table(parquet).to_pylist()
    assert len(stored) == len(rows)
    assert [s["audio"]["bytes"] for s in stored] == [r["audio"]["bytes"] for r in rows]
