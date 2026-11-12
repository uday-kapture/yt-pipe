# youtube-transcribe

Turn public YouTube podcasts into a private HF dataset of transcribed conversations without filling the disk:

```
sources.toml ─► discover ─► catalog.jsonl ─► run (repeats per batch, ~10 h of audio each)
                                              1. fetch      yt-dlp best audio → 16 kHz mono FLAC, SponsorBlock regions,
                                                            ≤ 55 min ASR chunks
                                              2. transcribe ../nemo-3.5-transcribe on the vLLM GPU server
                                              3. rows       subset `chunks`: one row per ≤ 55 min chunk (audio + full annotation)
                                                            subset `segments`: one row per ≤ 20 s speaker turn (clip + text + label)
                                              4. upload     one commit per batch to a private HF dataset
                                              5. delete     the batch's local files
                                              Google Chat: per-batch progress, peak storage, every error
```

This is Stage 0 of the VAP dataset pipeline in [PLAN.md](PLAN.md) (revised from `../plan.md` for YouTube sources).

## Setup

```bash
uv sync
```

- **ffmpeg** must be on `PATH` (`brew install ffmpeg`).
- yt-dlp needs a JavaScript runtime for YouTube. `yt-dlp[deno]` installs one into the venv; nothing else to do.
- A running **Nemotron 3.5 Transcribe** server: see `../nemo-3.5-transcribe/scripts/serve.sh` (GPU host).
- Secrets come only from the environment:

| Variable | Needed by | What |
| --- | --- | --- |
| `HF_TOKEN` | `run` | HF write token (or run `uv run hf auth login` once) |
| `GCHAT_WEBHOOK_URL` | `run` | Google Chat incoming-webhook URL. Unset = messages only go to the log |

## Usage

```bash
cp sources.example.toml sources.toml          # list your channels/playlists

uv run youtube-transcribe discover --config sources.toml --out data/
uv run youtube-transcribe run --config sources.toml --out data/ \
    --server http://gpu-host:8000 --hf-repo your-org/yt-podcasts --batch-hours 10

# Gold-set episodes: fetch with ≤360p video and keep everything locally (no upload, no delete)
uv run youtube-transcribe download --config sources.toml --out data/ --video --only gold_ids.txt
```

`run` is resumable: stop it any time and run the same command again (see below).

Sources: use a channel's `/videos` tab (newest first), a playlist, or a `/podcasts` tab (its playlists
are opened). `max_episodes` takes the first N that pass the filters, in listing order; playlists are
often oldest first.

## Outputs

**HF dataset (private), one repo, two subsets.** Each batch is one commit with three files:

| File | Subset | One row is |
| --- | --- | --- |
| `data/chunks/batch-NNNNN.parquet` | `chunks` (default, master copy) | ≤ 55 min chunk: FLAC audio, `transcript`, `annotation` JSON (IPUs with word timings and turn-taking labels, events, styles, diarization), episode metadata |
| `data/segments/batch-NNNNN.parquet` | `segments` | ≤ 20 s speaker turn: FLAC clip, `text`, `onset` label, `overlap` flag, episode metadata |
| `meta/batch-NNNNN.jsonl` | (not a subset) | one episode: `episode.json` (YouTube metadata, chosen audio format, SponsorBlock regions, chunk offsets) and the raw Transcribe responses, so labels can be recomputed without the GPU |

`load_dataset(repo, "chunks")` or `load_dataset(repo, "segments")`. The full column reference and the
reasoning behind the two subsets are in the dataset card, [dataset_card.md](src/youtube_transcribe/dataset_card.md),
which is uploaded as the repo's README the first time (later edits on HF are kept).
HF storage is about 120 MB per audio hour (both subsets together).

**Local `data/`:** `catalog.jsonl`, `state/batches.jsonl`, and `work/` while a batch is in flight (empty
otherwise). `download` keeps `episodes/<id>/` (original audio, `audio_16k.flac`, video, `episode.json`)
and `asr_input/<id>/c000.flac …`, which you can transcribe directly with
`uv run --project ../nemo-3.5-transcribe nemo-3-5-transcribe run data/asr_input --out data/asr --server …`.

## Google Chat messages

- run started / finished, with totals
- after each batch: videos transcribed and HF rows uploaded (batch and run totals), failures, peak
  storage this batch and this run, free disk
- immediately: every download, transcription or alignment failure; a stop (5 downloads failed in a row,
  server unreachable, upload failed) with where the local files are; Ctrl-C or `kill`; a crash

## If a run stops in the middle

Nothing is lost and nothing is uploaded twice. Run the same command again:

| Stopped during | On the next run |
| --- | --- |
| download | finished episodes are kept (`episode.json` marks them); a half-downloaded file resumes |
| transcription | only requests without a saved response are sent again |
| building parquet / upload | rebuilt from saved responses and uploaded again; files already on HF are not re-sent, and an unchanged upload makes no new commit |
| after the upload, before the ledger | the batch is uploaded again to the same paths, so no duplicate rows |
| after the ledger, before deleting | the leftover work dir is deleted |

A batch's local files are deleted only after its commit succeeded and it was written to
`data/state/batches.jsonl`. Episodes that failed are skipped for the rest of a run and retried by the next.
A hard kill (power loss, `kill -9`) sends no Chat message; the missing per-batch message is the sign.
Don't start two runs on the same `--out` directory at once.

## Caveats

- **Speaker ids reset per chunk**: episodes over 55 min are transcribed in pieces, so `speaker` 0 in
  `c000` and `c001` may be different people.
- **The ledger is local.** If `data/state/` is lost, the next run no longer knows which episodes are done: it
  downloads and transcribes them again and uploads them as new batches. Nothing on HF is overwritten
  (numbering continues after the last batch in `meta/`), but those rows are duplicates; drop them by
  `chunk_id`. Back up `data/state/batches.jsonl`.
- **YouTube can block you.** Downloads are sequential with a 5 s pause, and the run stops after 5
  failures in a row. If YouTube asks you to sign in, set `cookies_file` in `sources.toml`. Update yt-dlp
  often (`uv lock --upgrade-package yt-dlp`).
- **Throughput**: downloading and GPU transcription take turns, so the GPU sits idle while a batch
  downloads.
- Memory: each subset's rows are held in RAM while its parquet file is written (~60 MB per audio hour,
  one subset at a time).

## Development

```bash
uv run pytest
uvx ruff check src tests && uvx ruff format src tests
```
