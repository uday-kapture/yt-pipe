"""Disk-bounded batch loop: fetch → transcribe → upload to one private HF dataset → delete.

The dataset has two subsets, written as one commit per batch:

- ``chunks`` (default): one row per ≤ 55 min ASR chunk (audio + full annotation). The master copy.
- ``segments``: one row per ≤ 20 s speaker segment (clip + text + labels), for ASR / voicebot training.

plus ``meta/batch-NNNNN.jsonl`` (episode metadata and raw Transcribe responses).

Each batch lives in ``<out>/work/batch-NNNNN/`` and is deleted only after its upload succeeded and it
was appended to ``<out>/state/batches.jsonl``. A run that stops keeps the work dir; the next run resumes
it first. Progress and errors go to Google Chat.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import signal
import time
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import soundfile
from datasets import Audio, Dataset, Features, List, Value
from huggingface_hub import HfApi
from nemo_3_5_transcribe.annotate import INLINE_ONSETS
from nemo_3_5_transcribe.cli import main as nemo_main

from .audio import SAMPLE_RATE
from .youtube import fetch_episode, write_json

log = logging.getLogger(__name__)

MAX_CONSECUTIVE_FAILURES = 5
UPLOAD_TRIES = 3
MAX_SEGMENT_S = 20.0  # longer turns are split at word boundaries
SEGMENT_PAD_S = 0.1  # audio kept before the first and after the last word of a segment
_AUDIO = {"bytes": Value("binary"), "path": Value("string")}  # cast to Audio in write_parquet
_EPISODE_COLUMNS = ["episode_id", "chunk_id", "video_id", "channel", "channel_id", "title", "upload_date", "source",
                    "lang_hint", "license"]  # fmt: skip
_TIMES = {"start": Value("float64"), "end": Value("float64"), "duration": Value("float64")}
SEGMENT_FEATURES = Features(
    {
        "audio": _AUDIO,
        **{name: Value("string") for name in ["text", "text_tagged", "onset", *_EPISODE_COLUMNS]},
        **_TIMES,
        "speaker": Value("int32"),
        "ipus": List(Value("int32")),
        "overlap": Value("bool"),
        "pass_agreement": Value("float64"),
    }
)
CHUNK_FEATURES = Features(
    {
        "audio": _AUDIO,
        **{name: Value("string") for name in ["transcript", "transcript_tagged", "annotation", *_EPISODE_COLUMNS]},
        **_TIMES,
        "num_speakers": Value("int32"),
        "truncated": Value("bool"),
        "pass_agreement": Value("float64"),
    }
)


class Stop(Exception):
    """Stop the run but keep the current work dir so the next run can resume it."""


def post_chat(text: str) -> None:
    """Send ``text`` to the Google Chat webhook in ``GCHAT_WEBHOOK_URL``; never raises."""
    url = os.environ.get("GCHAT_WEBHOOK_URL")
    if not url:
        log.info("chat (NOT sent, GCHAT_WEBHOOK_URL is unset): %s", text)
        return
    log.info("chat: %s", text)
    request = urllib.request.Request(
        url,
        data=json.dumps({"text": text[:4000]}).encode(),
        headers={"Content-Type": "application/json; charset=UTF-8"},
    )
    try:
        urllib.request.urlopen(request, timeout=15).close()
    except Exception as error:  # a chat outage must not stop the pipeline
        log.warning("chat post failed: %r", error)
    time.sleep(1)  # stay under the webhook's rate limit


def alert(what: str, error: object) -> None:
    post_chat(f"⚠️ {what}: {str(error)[:1500]}")


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def dir_bytes(path: Path) -> int:
    """Bytes on disk under ``path``, counting hard-linked files once."""
    sizes = {}
    for file in path.rglob("*"):
        if file.is_file():
            stat = file.stat()
            sizes[(stat.st_dev, stat.st_ino)] = stat.st_size
    return sum(sizes.values())


def next_batch(catalog: list[dict], skip: set[str], batch_hours: float) -> list[dict]:
    """Catalog rows not in ``skip``, in order, up to ``batch_hours`` of audio (always at least one)."""
    batch, seconds = [], 0.0
    for row in catalog:
        if row["episode_id"] in skip:
            continue
        if batch and seconds + row["duration"] > batch_hours * 3600:
            break
        batch.append(row)
        seconds += row["duration"]
    return batch


def _episode_fields(chunk: dict, episode: dict) -> dict:
    youtube, source = episode["youtube"], episode["source"]
    return {
        "episode_id": episode["episode_id"],
        "chunk_id": chunk["id"],
        "video_id": youtube["id"],
        "channel": youtube.get("channel"),
        "channel_id": youtube.get("channel_id"),
        "title": youtube.get("title"),
        "upload_date": youtube.get("upload_date"),
        "source": source["name"],
        "lang_hint": source.get("lang_hint"),
        "license": youtube.get("license"),
    }


def chunk_record(conversation: dict, chunk: dict, episode: dict) -> dict:
    """One chunks-repo row (without audio) from a nemo conversation."""
    return {
        **_episode_fields(chunk, episode),
        "start": chunk["start"],
        "end": chunk["end"],
        "duration": round(chunk["end"] - chunk["start"], 3),
        "transcript": conversation["transcript"],
        "transcript_tagged": conversation["transcript_tagged"],
        # times inside are seconds from the chunk start; the local audio path is dropped
        "annotation": json.dumps({k: v for k, v in conversation.items() if k != "audio"}, ensure_ascii=False),
        "num_speakers": len(conversation["speakers"]),
        "truncated": conversation["truncated"],
        "pass_agreement": conversation["pass_agreement"],
    }


def speaker_turns(ipus: list[dict]) -> list[list[dict]]:
    """Group a chunk's IPUs into turns: an IPU labelled ``continue`` extends its speaker's last floor turn.

    Every other onset (switch, interruption, backchannel, ...) starts a new turn, so a listener's
    "haan" stays a row of its own and never merges into anyone's turn. Backchannels don't take the
    floor, so they never become the turn that a later ``continue`` extends.
    """
    turns: list[list[dict]] = []
    floor_turn: dict[int, list[dict]] = {}  # per speaker: their latest turn that held the floor
    for ipu in ipus:
        turn = floor_turn.get(ipu["speaker"])
        if ipu["onset"] == "continue" and turn is not None:
            turn.append(ipu)
            continue
        turns.append([ipu])
        if ipu["onset"] != "backchannel":
            floor_turn[ipu["speaker"]] = turns[-1]
    return turns


def turn_segments(turn: list[dict], all_ipus: list[dict], chunk_duration: float) -> list[dict]:
    """Split one turn into ≤ ``MAX_SEGMENT_S`` pieces at word boundaries; offsets are chunk seconds."""
    # (text, tagged text, start, end, ipu id) per word; an IPU's trailing tags ride on its last word
    words = []
    for ipu in turn:
        last = len(ipu["words"]) - 1
        for i, (text, start, end) in enumerate(ipu["words"]):
            tagged = ipu["word_groups"][i] + (f" {ipu['trailing_tags']}" if i == last and ipu["trailing_tags"] else "")
            words.append((text, tagged, start, end, ipu["id"]))
    pieces, first = [], 0
    for i in range(1, len(words) + 1):
        if i == len(words) or words[i][3] - words[first][2] > MAX_SEGMENT_S:
            pieces.append(words[first:i])
            first = i

    speaker, onset = turn[0]["speaker"], turn[0]["onset"]
    segments = []
    for n, piece in enumerate(pieces):
        start = max(0.0, piece[0][2] - SEGMENT_PAD_S)
        end = min(chunk_duration, piece[-1][3] + SEGMENT_PAD_S)
        tagged = [w[1] for w in piece]
        if n == 0 and onset in INLINE_ONSETS:
            tagged.insert(0, f"<int:{onset}>")
        segments.append(
            {
                "offset": round(start, 3),
                "duration": round(end - start, 3),
                "text": " ".join(w[0] for w in piece),
                "text_tagged": " ".join(tagged),
                "speaker": speaker,
                "onset": onset if n == 0 else "continue",
                # another speaker is audible somewhere in the clip
                "overlap": any(o["speaker"] != speaker and o["start"] < end and start < o["end"] for o in all_ipus),
                "ipus": sorted({w[4] for w in piece}),
            }
        )
    return segments


def segment_record(seg: dict, chunk: dict, episode: dict, pass_agreement: float | None) -> dict:
    """One segments row (without audio) from a ``turn_segments`` piece; times move to episode time."""
    start = round(chunk["start"] + seg["offset"], 3)
    return {
        **_episode_fields(chunk, episode),
        "text": seg["text"],
        "text_tagged": seg["text_tagged"],
        "speaker": seg["speaker"],
        "onset": seg["onset"],
        "overlap": seg["overlap"],
        "start": start,
        "end": round(start + seg["duration"], 3),
        "duration": seg["duration"],
        "ipus": seg["ipus"],
        "pass_agreement": pass_agreement,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# shortcut: a batch's rows sit in memory (~60 MB per audio hour each for chunks and segments, built one after the
# other); write row groups incrementally if batches grow past ~20 h
def _transcribed_chunks(asr: Path, episodes: dict[str, dict]):
    """(conversation, chunk, episode) for every chunk nemo annotated successfully."""
    chunks = {c["id"]: (c, ep) for ep in episodes.values() for c in ep["asr_chunks"]}
    for path in sorted((asr / "conversations").glob("*.json")):
        conversation = json.loads(path.read_text())
        if conversation["status"] == "ok" and conversation["id"] in chunks:
            yield conversation, *chunks[conversation["id"]]


def _chunk_rows(asr: Path, episodes: dict[str, dict]) -> list[dict]:
    """One row per transcribed chunk: its FLAC file as-is plus the full annotation."""
    rows = []
    for conversation, chunk, episode in _transcribed_chunks(asr, episodes):
        row = chunk_record(conversation, chunk, episode)
        row["audio"] = {"bytes": Path(conversation["audio"]).read_bytes(), "path": f"{chunk['id']}.flac"}
        rows.append(row)
    return rows


def _segment_rows(asr: Path, episodes: dict[str, dict]) -> list[dict]:
    """One row per speaker turn (or ≤ 20 s piece of one) with its FLAC clip."""
    rows = []
    for conversation, chunk, episode in _transcribed_chunks(asr, episodes):
        data, rate = soundfile.read(conversation["audio"], dtype="int16")  # one chunk (≤ 55 min) at a time
        chunk_duration = conversation.get("duration") or len(data) / rate
        for turn in speaker_turns(conversation["ipus"]):
            for seg in turn_segments(turn, conversation["ipus"], chunk_duration):
                row = segment_record(seg, chunk, episode, conversation["pass_agreement"])
                clip = io.BytesIO()
                first = int(seg["offset"] * rate)
                soundfile.write(clip, data[first : first + int(seg["duration"] * rate)], rate, format="FLAC")
                row["audio"] = {
                    "bytes": clip.getvalue(),
                    "path": f"{chunk['id']}_{seg['ipus'][0]}_{row['start']:.2f}.flac",
                }
                rows.append(row)
    return rows


def write_parquet(rows: list[dict], features: Features, path: Path) -> None:
    """Parquet with an HF ``Audio`` column, so the dataset viewer plays the audio."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Audio(...) in from_list would re-encode through torch + torchcodec; casting the stored bytes does not
    dataset = Dataset.from_list(rows, features=features).cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dataset.to_parquet(str(path))  # datasets keeps row groups ≤ ~100 MB, i.e. 1–2 chunks each


def _meta_lines(asr: Path, episodes: dict[str, dict]) -> str:
    """Per episode: its metadata plus the raw Transcribe responses, so labels can be recomputed without the GPU."""
    raw: dict[str, dict] = defaultdict(dict)
    for path in (asr / "raw").glob("*.json"):
        chunk_id, pass_name, _ = path.name.rsplit(".", 2)
        raw[chunk_id][pass_name] = json.loads(path.read_text())
    return "".join(
        json.dumps(
            {"episode": ep, "raw": {c["id"]: raw.get(c["id"], {}) for c in ep["asr_chunks"]}}, ensure_ascii=False
        )
        + "\n"
        for ep in episodes.values()
    )


def _prepare_repo(api: HfApi, repo: str) -> None:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type="dataset").private:  # exist_ok keeps an existing repo's visibility
        raise Stop(f"{repo} is public; refusing to upload. Make it private or pick another --hf-repo")
    if not api.file_exists(repo, "README.md", repo_type="dataset"):
        card = Path(__file__).with_name("dataset_card.md").read_text().replace("{repo}", repo)
        api.upload_file(
            path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
            commit_message="Add dataset card",
        )  # fmt: skip


def _upload(api: HfApi, repo: str, folder: Path, message: str) -> None:
    for attempt in range(UPLOAD_TRIES):
        try:
            # re-uploading unchanged files after a resume makes no new commit
            api.upload_folder(repo_id=repo, folder_path=str(folder), repo_type="dataset", commit_message=message)
            return
        except Exception as error:
            if attempt == UPLOAD_TRIES - 1:
                raise Stop(f"upload to {repo} failed: {error}") from error
            log.warning("upload attempt %d to %s failed: %r", attempt + 1, repo, error)
            time.sleep(30 * (attempt + 1))


def _check_server(server: str) -> None:
    try:
        urllib.request.urlopen(f"{server.rstrip('/')}/v1/models", timeout=15).close()
    except Exception as error:
        raise Stop(f"transcription server {server} unreachable: {error}") from error


def run_batch(n: int, rows: list[dict], work: Path, server: str, repo: str, api: HfApi, cookies: str | None) -> dict:
    """Fetch, transcribe and upload one batch; returns its ledger entry. Raises Stop to pause the run."""
    started = time.monotonic()
    peak = 0
    failures = 0

    def measure() -> None:
        nonlocal peak
        peak = max(peak, dir_bytes(work))

    episodes: dict[str, dict] = {}
    consecutive = 0
    for row in rows:
        try:
            episodes[row["episode_id"]] = fetch_episode(row, work / "episodes", work / "asr_input", cookies)
        except Exception as error:
            failures += 1
            consecutive += 1
            # keep a half-fetched episode's chunks out of transcription
            shutil.rmtree(work / "asr_input" / row["episode_id"], ignore_errors=True)
            alert(f"download failed {row['episode_id']} {row['url']}", error)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                raise Stop(f"{consecutive} downloads failed in a row (rate limit or bot check?)") from error
            continue
        consecutive = 0
        measure()
        ep_dir = work / "episodes" / row["episode_id"]
        for redundant in [*ep_dir.glob("audio_orig.*"), ep_dir / "audio_16k.flac"]:
            redundant.unlink(missing_ok=True)  # the ASR chunks are the audio that gets uploaded

    asr = work / "asr"
    if episodes:
        _check_server(server)
        failures_path = asr / "transcribe_failures.json"
        failures_path.unlink(missing_ok=True)  # don't re-report a resumed batch's old failures
        nemo_main(["run", str(work / "asr_input"), "--out", str(asr), "--server", server])
        measure()
        if failures_path.exists():
            for key, error in json.loads(failures_path.read_text()).items():
                failures += 1
                alert(f"transcription failed {key}", error)
    for conversation in _read_jsonl(asr / "manifests/conversations.jsonl"):
        if conversation["status"] != "ok":
            failures += 1
            alert(f"alignment failed {conversation['conversation_id']}", conversation["status"])

    name = f"batch-{n:05d}"
    upload = work / "upload"
    chunk_rows = _chunk_rows(asr, episodes)
    transcribed = sorted({r["episode_id"] for r in chunk_rows})
    n_chunks, hours = len(chunk_rows), round(sum(r["duration"] for r in chunk_rows) / 3600, 2)
    if chunk_rows:
        write_parquet(chunk_rows, CHUNK_FEATURES, upload / "data" / "chunks" / f"{name}.parquet")
    del chunk_rows  # free the batch's audio before cutting segment clips
    segment_rows = _segment_rows(asr, episodes)
    n_segments = len(segment_rows)
    if segment_rows:
        write_parquet(segment_rows, SEGMENT_FEATURES, upload / "data" / "segments" / f"{name}.parquet")
    del segment_rows
    if episodes:
        meta = upload / "meta" / f"{name}.jsonl"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(_meta_lines(asr, episodes))
        measure()
        # one commit for both subsets, so they never disagree about which batches are in
        _upload(api, repo, upload, f"{name}: {len(transcribed)} episodes, {n_chunks} chunks, {n_segments} segments")

    return {
        "batch": n,
        "episodes": sorted(episodes),
        "transcribed": transcribed,
        "chunks": n_chunks,
        "segments": n_segments,
        "hours": hours,
        "peak_bytes": peak,
        "failures": failures,
        "minutes": round((time.monotonic() - started) / 60, 1),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def last_batch_on_hub(files: list[str]) -> int:
    """Highest batch number already in the repo, so a lost local ledger can't overwrite earlier batches."""
    numbers = [int(m[1]) for f in files if (m := re.fullmatch(r"meta/batch-(\d+)\.jsonl", f))]
    return max(numbers, default=0)


def _totals(ledger: list[dict]) -> tuple[int, int, float, int]:
    """(videos transcribed, chunk rows, audio hours, segment rows) over the whole ledger."""
    return (
        sum(len(b["transcribed"]) for b in ledger),
        sum(b["chunks"] for b in ledger),
        round(sum(b["hours"] for b in ledger), 1),
        sum(b["segments"] for b in ledger),
    )


def run(catalog: list[dict], out: Path, server: str, repo: str, batch_hours: float, cookies: str | None) -> int:
    """Process the catalog batch by batch until done; returns the exit code."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # `kill` gets the same Chat notice as Ctrl-C
    if not os.environ.get("GCHAT_WEBHOOK_URL"):
        log.warning("GCHAT_WEBHOOK_URL is not set: progress and errors go to this log only, not to Google Chat")
    ledger_path = out / "state" / "batches.jsonl"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    work_root = out / "work"
    run_peak = 0
    failed: set[str] = set()  # skipped for the rest of this run; retried by the next run
    work = None
    try:
        api = HfApi()
        _prepare_repo(api, repo)
        hub_last = last_batch_on_hub(api.list_repo_files(repo, repo_type="dataset"))
        ledger = _read_jsonl(ledger_path)
        done = {e for b in ledger for e in b["transcribed"]}
        todo = len({r["episode_id"] for r in catalog} - done)
        post_chat(f"▶️ run started: {todo} of {len(catalog)} catalog episodes to go → {repo} (private)")

        while True:
            ledger = _read_jsonl(ledger_path)
            done = {e for b in ledger for e in b["transcribed"]}
            leftover = sorted(work_root.glob("batch-*/batch.json"))
            if leftover:
                work = leftover[0].parent
                n, rows = int(work.name.removeprefix("batch-")), json.loads(leftover[0].read_text())
                if any(b["batch"] == n for b in ledger):  # finished before a crash, only the delete was missed
                    shutil.rmtree(work)
                    continue
            else:
                rows = next_batch(catalog, done | failed, batch_hours)
                if not rows:
                    break
                n = max(max((b["batch"] for b in ledger), default=0), hub_last) + 1
                work = work_root / f"batch-{n:05d}"
                work.mkdir(parents=True)
                write_json(work / "batch.json", rows)

            entry = run_batch(n, rows, work, server, repo, api, cookies)
            with ledger_path.open("a") as ledger_file:
                ledger_file.write(json.dumps(entry) + "\n")
            failed |= {r["episode_id"] for r in rows} - set(entry["transcribed"])
            shutil.rmtree(work)
            work = None

            run_peak = max(run_peak, entry["peak_bytes"])
            ledger.append(entry)
            videos, chunks, hours, segments = _totals(ledger)
            post_chat(
                f"✅ batch {n:05d} done in {entry['minutes']} min\n"
                f"videos transcribed: {len(entry['transcribed'])} this batch · {videos} total "
                f"(catalog: {len(catalog)})\n"
                f"HF rows this batch: {entry['chunks']} chunks ({entry['hours']} h) · {entry['segments']:,} segments\n"
                f"HF rows total: {chunks:,} chunks ({hours} h) · {segments:,} segments → {repo}\n"
                f"failures this batch: {entry['failures']}\n"
                f"storage: peak {gb(entry['peak_bytes'])} this batch · max {gb(run_peak)} this run · "
                f"{gb(shutil.disk_usage(out).free)} free"
            )
    except Stop as stop:
        post_chat(f"🛑 run stopped: {stop}\nLocal files kept in {work or out}; re-run the same command to resume.")
        return 1
    except KeyboardInterrupt:
        post_chat(f"⏹️ run interrupted (Ctrl-C or kill). Local files kept in {work or out}; re-run to resume.")
        raise
    except Exception as error:
        post_chat(f"💥 run crashed: {type(error).__name__}: {str(error)[:1500]}\nLocal files kept in {work or out}.")
        raise

    videos, chunks, hours, segments = _totals(_read_jsonl(ledger_path))
    post_chat(
        f"🏁 run finished: {videos} videos transcribed · {chunks:,} chunks ({hours} h) · {segments:,} segments; "
        f"{len(failed)} episodes failed this run (retried next run); max storage this run {gb(run_peak)}"
    )
    return 0
