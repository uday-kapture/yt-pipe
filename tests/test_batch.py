import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile

from youtube_transcribe.batch import (
    CHUNK_FEATURES,
    SEGMENT_FEATURES,
    _chunk_rows,
    _segment_rows,
    last_batch_on_hub,
    next_batch,
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


@pytest.fixture
def nemo_output(tmp_path):
    """Fake nemo outputs: chunk c001 (10 s, starting 3291.4 s into the episode) is ok, c002 failed alignment."""
    chunk_path = tmp_path / "c001.flac"
    noise = np.random.default_rng(0).standard_normal(10 * RATE) * 1000
    soundfile.write(str(chunk_path), noise.astype("int16"), RATE)
    asr = tmp_path / "asr"
    (asr / "manifests").mkdir(parents=True)
    (asr / "conversations").mkdir()
    conversation = {"id": "yt_abc__c001", "audio": str(chunk_path), "status": "ok", "transcript": "<spk:0> achha",
                    "transcript_tagged": "<spk:0> achha", "speakers": {"0": {}}, "truncated": False,
                    "pass_agreement": 0.9, "ipus": [{"id": 0, "start": 2.0, "end": 4.9}]}  # fmt: skip
    (asr / "conversations" / "yt_abc__c001.json").write_text(json.dumps(conversation))
    failed = {"id": "yt_abc__c002", "audio": str(chunk_path), "status": "alignment_failed:no_words"}
    (asr / "conversations" / "yt_abc__c002.json").write_text(json.dumps(failed))
    seg = {"audio_filepath": str(chunk_path), "offset": 2.0, "duration": 3.0, "text": "achha", "text_tagged": "achha",
           "speaker": 0, "conversation_id": "yt_abc__c001", "ipu": 0, "onset": "start", "overlap": False}  # fmt: skip
    (asr / "manifests" / "segments.jsonl").write_text(json.dumps(seg) + "\n")
    (asr / "manifests" / "conversations.jsonl").write_text(
        json.dumps({"conversation_id": "yt_abc__c001", "pass_agreement": 0.9}) + "\n"
    )
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


def test_segment_rows_use_episode_time(nemo_output):
    asr, episodes, _ = nemo_output
    [segment] = _segment_rows(asr, episodes)
    assert (segment["start"], segment["end"]) == (3293.4, 3296.4)
    assert (segment["video_id"], segment["lang_hint"], segment["pass_agreement"]) == ("abc", "hi-en", 0.9)
    clip, clip_rate = soundfile.read(io.BytesIO(segment["audio"]["bytes"]))
    assert (len(clip), clip_rate) == (3 * RATE, RATE)
    assert set(SEGMENT_FEATURES) == set(segment)


@pytest.mark.parametrize(("build", "features"), [(_chunk_rows, CHUNK_FEATURES), (_segment_rows, SEGMENT_FEATURES)])
def test_parquet_has_a_playable_audio_column(nemo_output, tmp_path, build, features):
    asr, episodes, _ = nemo_output
    rows = build(asr, episodes)
    parquet = tmp_path / "subset.parquet"
    write_parquet(rows, features, parquet)

    schema = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"]
    assert schema["audio"]["_type"] == "Audio"  # what the HF viewer needs to play it
    [stored] = pq.read_table(parquet).to_pylist()
    assert stored["audio"]["bytes"] == rows[0]["audio"]["bytes"]
