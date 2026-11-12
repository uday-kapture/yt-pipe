---
pretty_name: YouTube podcast conversations
language:
- hi
- en
task_categories:
- automatic-speech-recognition
tags:
- audio
- speech
- conversation
- turn-taking
- podcasts
configs:
- config_name: chunks
  default: true
  data_files:
  - split: train
    path: data/chunks/*.parquet
- config_name: segments
  data_files:
  - split: train
    path: data/segments/*.parquet
---

# YouTube podcast conversations

Conversational audio from public YouTube podcasts (mostly Hindi, English and Hinglish), transcribed with
[NVIDIA Nemotron 3.5 Transcribe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B).
Speakers are attributed and words timestamped, and every turn carries a turn-taking label (backchannel,
interruption, overlap, …).

The same data is in two subsets:

| Subset | One row is | Use it for |
| --- | --- | --- |
| `chunks` (default) | up to 55 min of continuous conversation: the audio plus the full annotation | diarization, voice activity projection (VAP), turn-taking research, anything that needs the whole conversation |
| `segments` | one speaker's turn, ≤ 20 s: the clip, its text and its turn-taking label | training ASR models and voicebots |

```python
from datasets import load_dataset

chunks = load_dataset("{repo}", "chunks", split="train", streaming=True)
segments = load_dataset("{repo}", "segments", split="train", streaming=True)
```

## Why `chunks` is the master copy

A chunk keeps the whole conversation: both speakers, the pauses between them, their overlaps and the
exact timing. Short single-speaker clips can always be cut from a chunk (see below), but a conversation
cannot be rebuilt from clips. So if the two subsets ever disagree, or you need a different segmentation,
go back to `chunks`. The `segments` subset is a convenience: the same audio, already cut into turns.

## `chunks` columns

| Column | |
| --- | --- |
| `audio` | the chunk, 16 kHz mono FLAC |
| `transcript`, `transcript_tagged` | whole chunk as `<spk:N>` turns; the tagged form adds `<int:*>` turn-taking, `<ev:*>` event and `<sty:*>` speaking-style tags |
| `annotation` | JSON string with everything else (below) |
| `start`, `end`, `duration` | where the chunk sits in the full episode, in seconds |
| `num_speakers`, `truncated`, `pass_agreement` | speakers found; whether the model hit its output limit; word agreement between two transcription passes (a cheap quality filter) |
| `episode_id`, `chunk_id`, `video_id` | ids; `episode_id` is `yt_<video_id>`, `chunk_id` is `<episode_id>__c000`, `__c001`, … |
| `channel`, `channel_id`, `title`, `upload_date`, `license`, `source`, `lang_hint` | episode metadata; `lang_hint` is set per source by hand |

`annotation` holds, with **all times in seconds from the start of the chunk** (add `start` for episode time):

- `ipus`: inter-pausal units, i.e. each speaker's stretches of speech. Each has `speaker`, `start`, `end`,
  `text`, `text_tagged`, `words` (`[text, start, end]`), its `onset` label, and the timing features
  behind the label (`target`, `gap_s`, `overlap_s`, `yield_s`, `cut_off_by`).
- `speakers`: per speaker, the full text, word count, speech time and IPU ids.
- `events` (laughter, breath, …) and `styles` (speaking-style spans), each attached to a speaker.
- `diarization`: the model's speaker segments.
- `stats`, `passes`: label counts and the raw text of each transcription pass.

Onset labels: `start`, `continue` (same speaker after a pause), `switch` (clean turn change),
`backchannel` (short feedback while the other speaker keeps the floor), `interruption` (starts inside
the other speaker's turn and that speaker stops), `failed_interruption`, `overlap_turn_end` (starts in
the last moments of the other turn) and `simultaneous_start`.

## `segments` columns

| Column | |
| --- | --- |
| `audio` | the clip, 16 kHz mono FLAC, ≤ 20 s (+0.1 s padding) |
| `text`, `text_tagged` | the turn's words; the tagged form adds `<int:*>`, `<ev:*>` and `<sty:*>` tags |
| `speaker`, `onset`, `overlap`, `ipu` | speaker index, turn-taking label, `true` if another speaker is audible in the clip (filter these out for clean single-speaker ASR), IPU id in the chunk's `annotation` |
| `start`, `end`, `duration` | in the **full episode**, in seconds |
| `pass_agreement`, ids and episode metadata | as in `chunks` |

## Cutting your own segments from `chunks`

```python
import io, json
import soundfile
from datasets import Audio, load_dataset

chunks = load_dataset("{repo}", "chunks", split="train", streaming=True)
chunks = chunks.cast_column("audio", Audio(decode=False))  # raw FLAC bytes; no torch needed
for row in chunks:
    samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]))
    for ipu in json.loads(row["annotation"])["ipus"]:
        clip = samples[int(ipu["start"] * rate) : int(ipu["end"] * rate)]
        episode_time = row["start"] + ipu["start"]
        # ipu["speaker"], ipu["text"], ipu["onset"], ipu["words"], ...
```

## `meta/`

`meta/batch-NNNNN.jsonl` has one line per episode:

- `episode`: YouTube metadata (title, description, chapters, license), the audio format that was
  downloaded, the SponsorBlock regions (sponsor reads, intros, outros, …), and where each chunk starts in
  the episode. Times are in episode seconds. Sponsor and intro regions are **marked, not removed**: the
  audio still contains them.
- `raw`: the model's raw responses per chunk, so labels can be recomputed with different thresholds
  without running the model again.

## How it was made

1. A curated list of podcast channels and playlists is expanded with yt-dlp, and episodes are filtered
   (15 min – 4 h, no live streams, trailers or clips).
2. The best audio track is converted to 16 kHz mono. Episodes over 55 min are split at the quietest
   moment near the 55 min mark, or inside a sponsor/intro region if one is there.
3. Each chunk is transcribed twice: once with speaker tags and word timestamps, and once with event and
   speaking-style tags. Turn-taking labels are derived from per-speaker word timing.
4. Batches of about 10 hours are uploaded as `data/chunks/batch-NNNNN.parquet`,
   `data/segments/batch-NNNNN.parquet` and `meta/batch-NNNNN.jsonl`.

## Caveats

- **Labels are automatic.** Speaker attribution and turn-taking labels come from the model's diarization
  and word timing. They are weakest for short overlapping speech, so short backchannels are under-counted.
- **Speaker numbers only hold within one chunk.** `speaker` 0 in `c000` and in `c001` of the same episode
  may be different people. A turn can also be split across a chunk boundary.
- **Edited audio.** Most podcasts are edited, so jump cuts shorten pauses and overlaps in places.
- **Source and use.** Episodes are public YouTube videos collected for internal research. The `license`
  column records YouTube's license field. Do not redistribute.
