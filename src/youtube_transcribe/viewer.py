"""Browse conversations in the HF dataset: read the transcript turn by turn, click a turn to hear it.

    HF_REPO=org/name HF_TOKEN=hf_... uv run --extra viewer yt-viewer     # then open http://127.0.0.1:7860

The repo comes from ``HF_REPO``; the token from ``HF_TOKEN`` (or your ``hf auth login``). Neither is
typed into the page. Rows are the ``chunks`` subset in the dataset viewer's order (row 0 = first row of
the first batch file). Only the requested rows are read; a conversation's audio is fetched when you open it.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pyarrow.parquet as pq
import soundfile
from huggingface_hub import HfFileSystem

from .batch import speaker_turns

CONTEXT_S = 2.0  # audio played before and after a clicked turn
TEXT_COLUMNS = ["episode_id", "chunk_id", "video_id", "title", "channel", "upload_date", "start", "duration",
                "num_speakers", "pass_agreement", "lang_hint", "annotation"]  # fmt: skip
_AUDIO_DIR = Path(tempfile.mkdtemp(prefix="yt-viewer-"))


def _repo() -> str:
    repo = os.environ.get("HF_REPO", "").strip()
    if not repo:
        raise gr.Error("HF_REPO is not set. Start the viewer with HF_REPO=org/name (and HF_TOKEN for a private repo).")
    return repo


def _fs() -> HfFileSystem:
    return HfFileSystem()  # token: HF_TOKEN, else the one from `hf auth login`


@lru_cache(maxsize=1)
def _row_groups(repo: str) -> list[tuple[str, int, int, int]]:
    """(file, row group, first row, rows) for the chunks subset, in row order."""
    fs = _fs()
    groups, row = [], 0
    for path in sorted(fs.glob(f"datasets/{repo}/data/chunks/*.parquet")):
        with fs.open(path, "rb") as f:
            meta = pq.ParquetFile(f).metadata
        for g in range(meta.num_row_groups):
            groups.append((path, g, row, meta.row_group(g).num_rows))
            row += meta.row_group(g).num_rows
    return groups


def read_row(repo: str, row: int, columns: list[str]) -> dict:
    groups = _row_groups(repo)
    for path, g, first, n in groups:
        if first <= row < first + n:
            with _fs().open(path, "rb") as f:
                table = pq.ParquetFile(f).read_row_group(g, columns=columns)
            return table.slice(row - first, 1).to_pylist()[0]
    total = groups[-1][2] + groups[-1][3] if groups else 0
    raise gr.Error(f"Row {row} doesn't exist: the chunks subset of {repo} has {total} rows (0–{total - 1}).")


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return (
        f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
        if seconds >= 3600
        else f"{seconds // 60}:{seconds % 60:02d}"
    )


def turn_table(annotation: dict, tagged: bool) -> tuple[list[list], list[tuple[float, float]]]:
    """Transcript rows (one per speaker turn) and each turn's (start, end) in chunk seconds."""
    rows, spans = [], []
    for turn in speaker_turns(annotation.get("ipus", [])):
        start, end = turn[0]["start"], turn[-1]["end"]
        text = " ".join(ipu["text_tagged" if tagged else "text"] for ipu in turn)
        rows.append([len(rows), f"{_clock(start)}–{_clock(end)}", f"S{turn[0]['speaker']}", turn[0]["onset"], text])
        spans.append((start, end))
    return rows, spans


def load(first_row: float, count: float):
    repo = _repo()
    try:
        conversations = [
            {"row": r, **read_row(repo, r, TEXT_COLUMNS)} for r in range(int(first_row), int(first_row + count))
        ]
    except FileNotFoundError as error:
        raise gr.Error(f"Can't read {repo}: wrong HF_REPO, or HF_TOKEN has no access ({error})") from error
    choices = [
        (f"row {c['row']} · {c['chunk_id']} · {_clock(c['duration'])} · {c['num_speakers']} speakers · {c['title']}", i)
        for i, c in enumerate(conversations)
    ]
    return conversations, gr.Dropdown(choices=choices, value=0 if choices else None)


def show(index: int | None, conversations: list[dict], tagged: bool):
    if index is None or not conversations:
        return "", [], [], None, None
    c = conversations[index]
    annotation = json.loads(c["annotation"])
    rows, spans = turn_table(annotation, tagged)
    onsets = Counter(r[3] for r in rows)
    info = (
        f"### {c['title']}\n"
        f"**{c['channel']}** · uploaded {c['upload_date']} · `{c['chunk_id']}` · "
        f"[open on YouTube at {_clock(c['start'])}](https://www.youtube.com/watch?v={c['video_id']}&t={int(c['start'])}s)\n\n"
        f"chunk {_clock(c['start'])}–{_clock(c['start'] + c['duration'])} of the episode · "
        f"{c['num_speakers']} speakers · "
        f"lang hint `{c['lang_hint']}` · pass agreement {c['pass_agreement']} · "
        f"{len(rows)} turns: " + ", ".join(f"{k} {v}" for k, v in onsets.most_common())
    )
    audio = _AUDIO_DIR / f"{c['chunk_id']}.flac"
    if not audio.exists():
        audio.write_bytes(read_row(_repo(), c["row"], ["audio"])["audio"]["bytes"])
    return info, rows, spans, str(audio), None


def play_turn(spans: list[tuple[float, float]], audio_path: str | None, evt: gr.SelectData):
    if not audio_path or not spans:
        return None
    start, end = spans[evt.index[0]]
    info = soundfile.info(audio_path)
    first = max(0, int((start - CONTEXT_S) * info.samplerate))
    data, rate = soundfile.read(audio_path, start=first, stop=int((end + CONTEXT_S) * info.samplerate), dtype="int16")
    return rate, data


def build() -> gr.Blocks:
    with gr.Blocks(title="Podcast dataset viewer") as demo:
        gr.Markdown(
            f"## Podcast dataset viewer · `{os.environ.get('HF_REPO', 'HF_REPO not set')}`\n"
            "Read a conversation turn by turn; **click a turn to hear it**."
        )
        conversations, spans = gr.State([]), gr.State([])
        with gr.Row():
            first_row = gr.Number(label="First row", value=0, minimum=0, precision=0, scale=1)
            count = gr.Number(label="How many", value=3, minimum=1, maximum=20, precision=0, scale=1)
            load_btn = gr.Button("Load", variant="primary", scale=1)
        with gr.Row():
            picker = gr.Dropdown(label="Conversation", choices=[], type="value", scale=4)
            tagged = gr.Checkbox(label="Show tags (<int:…> <ev:…> <sty:…>)", value=False, scale=1)
        info = gr.Markdown()
        full_audio = gr.Audio(label="Whole conversation", type="filepath", interactive=False)
        turn_audio = gr.Audio(label="Clicked turn (±2 s)", autoplay=True, interactive=False)
        transcript = gr.Dataframe(
            headers=["#", "time", "speaker", "onset", "text"], datatype=["number", "str", "str", "str", "str"],
            interactive=False, wrap=True, max_height=700, column_widths=["4%", "11%", "7%", "13%", "65%"],
        )  # fmt: skip

        load_btn.click(load, [first_row, count], [conversations, picker])
        for event in (picker.change, tagged.change):
            event(show, [picker, conversations, tagged], [info, transcript, spans, full_audio, turn_audio])
        transcript.select(play_turn, [spans, full_audio], turn_audio)
    return demo


def main() -> None:
    if not os.environ.get("HF_REPO", "").strip():
        sys.exit("HF_REPO is not set. Run: HF_REPO=org/name HF_TOKEN=hf_... uv run --extra viewer yt-viewer")
    build().launch(server_name="127.0.0.1")  # local only: the token never leaves this machine except to HF


if __name__ == "__main__":
    main()
