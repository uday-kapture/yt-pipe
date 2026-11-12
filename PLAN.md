# VAP Evaluation Dataset from YouTube Podcasts — Pipeline Plan

> Revised from `../plan.md` (self-recorded episodes). The source is now **public YouTube podcasts** pulled with
> yt-dlp from a curated channel list, which gives far more recordings. Later stages are unchanged except where
> the new source changes them. Stage 0 (acquisition, batching, HF upload, Google Chat reporting) is built in this
> project; see `README.md`.

## 1. Goal

Build a dataset for **evaluating and fine-tuning** Voice Activity Projection (VAP) models from public YouTube podcast episodes (Hindi/English, mostly edited multi-camera shows). For every episode, produce:

- **Who is speaking when** — a per-speaker voice-activity matrix at VAP frame rate.
- **Transcripts** — word-level, speaker-attributed, timestamped.
- **Turn-taking events** — backchannel (with affirmation as an attribute), interruption, barge-in, plus supporting classes.
- **Removable tags** — events live in a separate layer, so a clean transcript can be rendered without them.

From these, build two sets: a hand-checked **gold set** used only for evaluation, and a larger automatically labeled **training set** for fine-tuning (Stage 11).

## 2. Constraints and assumptions

### Known constraints

| Constraint | Consequence |
|---|---|
| No per-speaker channels; only the final mixed audio | Speaker activity must be **estimated** by diarization; short overlapped speech (backchannels) is the weakest point |
| No music files, project files, or unedited recordings | Edits are baked in; cuts and music must be **detected**, not read from a source |
| Speaker count varies a lot per episode | VAP is two-party, so multi-party talk needs special handling; diarization quality varies with speaker count |
| Conversations mix Hindi and English | Code-switching affects transcription, backchannel word lists, and sentence-completion judgment |
| Many channels, unknown speakers | No fixed cast with reference clips; hosts recur within a channel, guests change per episode |
| Editing style varies per channel | Cut density, music beds and intros differ by channel; detectors and statistics are kept per channel |
| Episodes often run longer than 1 hour | Transcribe takes ≤ 1 h per request, so long episodes are split into chunks at quiet points (Stage 0) |
| YouTube rate limits and bot checks; yt-dlp breaks when YouTube changes | Downloads are sequential with pauses, stop after repeated failures, and yt-dlp is kept up to date |
| Disk is limited; the corpus can grow without bound | Work runs in batches: download → transcribe → upload to HF → delete local files |
| No per-speaker ground truth | Pipeline loss is measured on proxy data and a hand-checked gold set |

### Scope decisions

- The dataset is used both to **evaluate** and to **fine-tune** VAP models. Training uses automatic labels; evaluation uses a held-out, hand-checked gold set.
- Sources are a **curated list** of channels/playlists (`sources.toml`). Any public video is allowed (internal research); the YouTube license is recorded per episode, not filtered.
- **Video** is fetched **on demand only** (≤ 360p, video-only), for gold-set episodes. It is used to:
  - tell real edit cuts apart from camera switches (Stage 2);
  - optionally, see who is visibly speaking during overlaps (Stage 3);
  - let annotators mark cuts while watching (Stage 10).
- The bulk run uploads to **one private HF dataset with two subsets** and then deletes local audio: `chunks` (≤ 55 min of continuous conversation with the full annotation; the master copy that later stages read) and `segments` (≤ 20 s single-speaker clips with text and turn-taking labels, for ASR/voicebot training).
- A **limited** amount of hand annotation is possible, so annotation effort is prioritized (Stage 10).

### Models

- **ASR:** Nemotron 3.5 Transcribe (`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`).
  - Offline only; recordings up to 1 hour; 40 languages.
  - Supports verbatim and speaker-attributed instructions, word timestamps via the CTC timestamp option, and a context list of expected terms.
  - Word timestamps are **not** available with the DFlash2 draft model; the SGLang-Omni serving route has no timestamps or speakers. Serve via vLLM/NeMo.
- **Diarization:** Nemotron 3 Diarization (Sortformer family).
  - Up to 8 speakers.
  - Output every 10 ms, upsampled from 80 ms internal frames, so **effective onset precision is about 80 ms**.
  - Speakers are numbered by first appearance, so labels reset every episode.
- **Why both:** Transcribe's speaker tags label *words*, but VAP needs *frame-level* activity for every speaker, including overlaps and sounds that produce no words. The standalone diarizer is the source of truth for who speaks when; Transcribe's tags are a cheap cross-check in Stage 6.
  - Activity without words (laughter, dropped backchannels) is invisible in word tags.
  - Word tags give each word one speaker; the diarizer can mark several speakers active in the same frame.
  - The diarizer gives per-frame probabilities for thresholds and confidence tiers; tags are hard decisions.
  - Transcribe runs in < 1 hour chunks, and its speaker numbering likely restarts per chunk; the diarizer's chunked inference keeps speakers consistent across a whole episode.
  - Transcribe uses the same diarizer internally, so the two aren't independent opinions on identity. The gain is the frame-level representation, not a second model.

## 3. Ground rules (apply to every stage)

1. **One timeline.** All times refer to the original episode file. Audio is never physically cut; regions are excluded by marking them.
2. **Separate layers.** Words, voice activity, segments, events, excluded regions, and cuts are stored as independent layers that reference each other by ID and time.
3. **Confidence and provenance everywhere.** Every word, region, and event records which tool produced it and how confident it is.
4. **Versioned config.** Every threshold, rule, and judge prompt lives in a versioned config. Changing a definition means re-running only the cheap downstream stages.
5. **Never score across edits.** No evaluation window may overlap a detected cut or an excluded region.

## 4. Pipeline overview

```
 Stage 0  Acquisition (yt-dlp), chunking, batched transcribe → HF upload → cleanup
    │
 Stage 1  Non-conversation regions (intros, outros, cold opens, music, SFX)
    │
 Stage 2  Cut detection
    │
 Stage 3  Diarization & voice activity
    │
 Stage 4  Speaker stream separation
    │
 Stage 5  Transcription (mix pass + separated-stream pass in overlaps)
    │
 Stage 6  Reconciliation (words ↔ voice activity)
    │
 Stage 7  Segmentation & floor tracking
    │
 Stage 8  Event detection & labeling
    │
 Stage 9  Storage, exports & VAP input construction
    │
 Stage 10 Gold set, pipeline-loss measurement & evaluation
    │
 Stage 11 Fine-tuning
```

---

## 5. Stages

### Stage 0 — Acquisition, chunking, and batched upload (built)

**Purpose:** a steady, disk-bounded supply of episodes with consistent inputs and metadata.

- **Discover:** expand each source (channel `/videos` tab, playlist, or `/podcasts` tab) with yt-dlp flat extraction. Filter out live streams, clips/shorts (duration bounds) and trailers/highlights (title regex). Dedupe by video ID. Output: `catalog.jsonl`.
- **Fetch:** best audio track (the chosen format is recorded, to catch auto-dubbed tracks), converted to 16 kHz mono FLAC, which is the episode's single timeline. Metadata: title, description, channel, upload date, license, chapters.
- **SponsorBlock** segments (`sponsor`, `selfpromo`, `interaction`, `intro`, `outro`, `preview`, `music_offtopic`) seed `excluded_regions` (method `sponsorblock`). `filler` and `hook` are not excluded; tangents and greetings are still talk.
- **ASR chunks:** episodes over 55 min are split for Transcribe's 1 h limit. Each cut falls in the last 2 min before the cap, inside an excluded region if one is there, otherwise at the quietest 100 ms frame. `asr_chunks[].start` maps chunk time back to episode time.
- **Batches:** about 10 h of audio per batch: fetch → transcribe and annotate (`nemo-3.5-transcribe`) → build rows for both subsets → one commit to a private HF dataset (`data/chunks/batch-N.parquet`, `data/segments/batch-N.parquet`, plus `meta/batch-N.jsonl` with each episode's metadata and raw Transcribe responses) → delete the batch's local files. A ledger (`state/batches.jsonl`) makes runs resumable.
- **Reporting:** Google Chat gets a message per batch (videos transcribed, HF rows uploaded, peak storage) and every error as it happens.
- **Gold-set episodes** are fetched separately and kept locally (`download --video --only gold.txt`).
- **Speaker reference material:** there is no fixed cast. Hosts are discovered per channel after diarization (Stage 3). Guests are episode-local speakers registered from their first long solo stretch.

**Output:** private HF dataset (`chunks` and `segments` subsets + per-episode metadata), `episode.json` per kept episode, batch ledger.

### Stage 1 — Non-conversation regions

**Purpose:** find intros, outros, cold opens, music, and sound effects before they confuse the diarizer and ASR. Music with vocals can appear as an extra "speaker"; ASR may transcribe lyrics or invent text over instrumental passages.

Run three detectors in order:

0. **SponsorBlock** (Stage 0) already marks sponsor reads, intros, outros, previews and non-music sections where the community has labeled them. Coverage of Hindi podcasts may be sparse, so the detectors below still run.
1. **Repeated-asset discovery, per channel.** No source files exist, but intros, outros, and transition stings repeat across a channel's episodes.
   - Fingerprint (Chromaprint / audfprint) the first and last few minutes of every episode in a channel against each other.
   - Segments that recur across episodes are your assets. Cut one clean copy of each from the episode where it is least obscured.
   - Use those copies as templates and match them across all episodes for exact start and end times.
   - Repeat the cross-episode matching on music-classified regions (from detector 3) to discover reused background tracks.
2. **Cold-open self-matching.** Fingerprint the first few minutes of each episode against the rest of the same episode. Segments that appear twice are teasers or recaps: exclude the teaser copy, keep the in-context one.
3. **General music/sound classifier** for everything else.
   - Candidates: inaSpeechSegmenter (speech/music/noise), or an AudioSet tagger (PANNs, BEATs) for per-frame music and sound-effect scores.
   - Smooth with a minimum region duration so short flickers don't become regions.
   - Ducked background music under speech is the hard case: use a lower music threshold inside speech regions and spot-check.

**Decisions by region type:**

| Region type | Action |
|---|---|
| Intro, outro, cold open, sponsor read, montage | Exclude |
| Music-only transition | Exclude, and pass to Stage 2 as a likely cut |
| Speech over background music | Tag `music_bed`; keep with vocal separation (Demucs vocals) **or** exclude — decided by the Stage 10 gold-set test |
| Sound effects / meme audio | Exclude |

**Output:** `excluded_regions` and `conditions` layers.
**Check:** listen to region boundaries on a sample; total excluded time per episode looks plausible.

### Stage 2 — Cut detection

**Purpose:** find edit points, because editing distorts pause and overlap timing, which is exactly what VAP measures. There is no edit list, so the cut list is entirely estimated.

**Sources:**

- Video shot boundaries (e.g. PySceneDetect), split into two kinds:
  - **Jump cut:** the framing before and after is nearly identical but the person visibly jumps. Strong evidence of an audio cut.
  - **Camera switch:** the shot changes to a different angle or person. In multi-camera edits these often follow whoever is talking while the audio stays continuous, so a camera switch alone is **not** a cut; it needs corroboration from the audio.
- Audio discontinuities: sudden changes in room tone, background level, or spectrum.
- Music transitions from Stage 1.

Merge into one cut list with a confidence per cut (higher when sources agree).

Video is only fetched for gold-set episodes, so most episodes rely on audio discontinuities and Stage 1 transitions. Keep cut statistics **per channel**. A channel's measured edit density (cuts per minute on its gold episodes) is a stratification variable and a filter for the training set.

**Policy:** err toward over-detecting. A false cut costs a little data; a missed cut silently corrupts timing.

**Padding rule:** an event is not scored if a cut falls in its preceding context or within the ~2 s prediction horizon after it. Start with a few seconds of padding; set the final value from how far estimated cut times drift from hand-marked ones (Stage 10).

**Output:** `cuts` layer.

### Stage 3 — Diarization and voice activity

**Purpose:** per-speaker voice activity, the foundation for VAP labels and events.

- Run Nemotron 3 Diarization on the audio left after Stage 1.
- **Keep raw per-frame probabilities**, not only thresholded segments.
- **Per-channel host discovery:** extract speaker embeddings (e.g. TitaNet or ECAPA-TDNN) for each diarized speaker and cluster them across a channel's episodes. Clusters that recur in many episodes are hosts (`<channel>_host1`, …); their cleanest long solo stretches become reference clips. Everyone else is an episode-local guest.
- **Boundary refinement:** where only one speaker is active, sharpen onsets and offsets with a fine-grained VAD (e.g. Silero). Inside overlaps there is nothing to refine against.
- **Second-speaker threshold:** tune a lower activation threshold for a second speaker during overlap, on the gold set, to recover short backchannels without inventing false ones.
- **Active-speaker count:** record the number of active participants per window; it is a key stratification variable.
- **Optional — visual active-speaker detection:** where faces are on screen, an audio-visual active-speaker model (TalkNet / Light-ASD style) can show who is talking during overlaps, exactly where audio-only diarization is weakest. Use it as a second opinion on short overlapped speech, not as the primary source; it only helps while the speaker is visible.

**Output:** voice-activity matrix at 20 ms frames (probabilities + decisions), speaker map.
**Known risk:** short overlapped speech is under-detected, which biases the event set toward louder, longer backchannels. Measured in Stage 10.

### Stage 4 — Speaker stream separation

**Purpose:** one audio stream per speaker, used for (a) a second ASR pass in overlaps and (b) VAP inputs for evaluation and training.

- Use target-speaker extraction guided by each speaker's reference clip.
- Run in windows; keep each stream on the right person across windows using speaker embeddings.
- **Do not** build VAP inputs by muting the mix wherever the diarizer says a speaker is silent. The diarizer would then decide both what the model hears and what counts as correct, so its errors would never show up in the evaluation. Separation keeps input and labels from different sources.

**Output:** per-speaker separated audio streams.

### Stage 5 — Transcription

**Purpose:** word-level, speaker-attributed, timestamped transcripts.

**Mixed-audio pass**, with these settings:
- **Prompt:** verbatim + speaker-attributed.
- **Timestamps:** CTC timestamp option on.
- **No DFlash2.**
- **Chunking:** done in Stage 0 (≤ 55 min, cut in excluded regions or at the quietest point). Transcribe's speaker numbers restart per chunk, so the diarizer remains the whole-episode source of truth for identity.
- **Context list:** host and guest names, recurring topics and terms, taken from the episode title, description and chapters.

**Overlap pass:** wherever the diarizer shows overlap, re-run ASR on the separated streams for that span to recover dropped backchannels.

**Code-switching:**
- **Script convention:** fix one convention before running anything (see Open decisions) and put it in the prompt.
- **Normalization:** keep the original text and add a normalized single-script form for matching and WER. Normalize spelling variants (acha / achha / accha).
- **Language tag:** each stretch of speech gets `en`, `hi`, or `mixed`.
- **Pilot first:** check code-switching behaviour on a few gold minutes before scaling. Watch for Hindi translated into English or script flips mid-sentence.

**Output:** `words` layer (original text, normalized text, speaker, times, language, source pass, confidence).

### Stage 6 — Reconciliation

**Purpose:** merge words with voice activity and resolve conflicts.

| Situation | Rule |
|---|---|
| Word outside any speaker's activity | Flag as likely error |
| Activity with no words | Non-lexical segment: laughter, breath, or missed backchannel; label with ASR rich tags or a laughter detector |
| Short word attributed to speaker A inside a brief burst of speaker B activity | Reassign to B, or flag |
| Mixed-audio and separated-stream passes disagree | Keep both versions; mark low confidence |

Use voice-activity boundaries for timing and ASR for word content; ASR timestamps are less precise at the edges of short words.

**Output:** reconciled `words` layer, `nonlexical` segments.

### Stage 7 — Segmentation and floor tracking

**Purpose:** structural units for event detection.

- **Speech stretches:** per speaker, merge speech separated by short gaps into continuous stretches (inter-pausal units, IPUs).
- **Derived units:** from IPUs, derive turns, within-speaker pauses, between-speaker gaps, and overlaps.
- **Floor-holder track:** at each moment, who holds the current turn. When unclear, use the speaker with the most recent long IPU. Needed because with 3+ people, "interrupted whom" is not obvious.

**Output:** `segments` layer and `floor` track.

### Stage 8 — Event detection and labeling

**Purpose:** label every moment someone starts talking while someone else holds the floor.

**Candidates:** every onset by speaker B while A is speaking, or during a pause before A's sentence is complete.

**Features per candidate:**

| Feature | Meaning |
|---|---|
| `onset_context` | `overlap` / `mid_sentence_pause` / `after_completion` |
| `a_yielded` | A stopped within a short window and did not resume |
| `a_cut_off` | A stopped mid-word or mid-phrase |
| `b_took_floor` | B kept talking beyond a short response |
| `b_duration`, `b_word_count` | Size of B's contribution |
| `b_function` | `continuer` / `agreement` / `other`, from the word list + judge |
| `n_active` | Number of active participants |
| `lang` | Language of B's words and of A's surrounding speech |

**Sentence-completion judge:** an LLM judge reads A's words up to B's onset and decides whether the sentence was complete.
- Run it on the existing MLflow LLM-as-judge setup so prompts are versioned.
- Hindi is verb-final: clauses usually aren't complete until the verb and auxiliary arrive ("hai", "tha", "raha hai"). Tags like "na?" and "hai na?" often invite a response.
- Give the judge Hinglish examples from the gold set, and measure its accuracy per language.

**Backchannel word list:** bilingual and grown from the gold set. Seed:
- **Continuers:** hmm, haan, haan haan, achha, ji, okay, yeah, right
- **Agreement:** sahi hai, bilkul, theek hai, exactly, true

Context still matters ("achha" can mean "oh, really?"), so send ambiguous cases to the judge.

**Labels:**

| Label | Definition |
|---|---|
| `backchannel` | B does not take the floor; A continues. Attribute `function`: `continuer` or `agreement` (agreement = "affirmation") |
| `interruption` | B starts before A's sentence is complete and takes the floor. Attribute `outcome`: `successful` (A yields) or `competitive` (both keep talking) |
| `barge_in` | Subtype of interruption where A stops abruptly (cut off mid-word or mid-phrase) |
| `cooperative_completion` | B finishes A's sentence |
| `laughter_overlap` | Overlap consisting of laughter |
| `smooth_shift` | B starts after A's sentence is complete — the comparison class |
| `unclassified` | Below the confidence floor; never forced into a class |

**Multi-party attribution:** every event records `by`, `over` (the floor holder), and `n_active`. A third person's backchannel during an A–B exchange is attributed to the floor holder by default; the judge can override when the content clearly addresses someone else (a name, a direct "tu" or "aap").

**Confidence per event:** combine the diarizer probability margin, whether it relied on overlap detection, distance to the nearest cut, and judge confidence.

**Output:** `events` layer with `rule_version` and judge prompt version.

### Stage 9 — Storage, exports, and VAP inputs

**Storage:** one JSON file per episode plus a compact array file for voice activity. Stage 0's `episode.json` is the seed of this file: it already carries `episode_id`, `versions`, source and YouTube metadata, `excluded_regions` and `asr_chunks`. In the bulk run it is uploaded to HF (`meta/batch-N.jsonl`) and deleted locally.

```json
{
  "episode_id": "ep012",
  "versions": {"pipeline": "0.3", "config": "cfg_v7", "asr": "nemotron-3.5-transcribe", "diarizer": "nemotron-3-diarization"},
  "speakers": [
    {"id": "podx_host1", "type": "host"},
    {"id": "ep012_guest1", "type": "episode_local"}
  ],
  "excluded_regions": [
    {"start": 0.0, "end": 38.2, "reason": "intro", "method": "template_match", "confidence": 0.98, "action": "exclude"}
  ],
  "conditions": [
    {"start": 905.1, "end": 962.7, "condition": "music_bed", "action": "separated"}
  ],
  "cuts": [
    {"t": 412.36, "sources": ["video_shot", "audio_discontinuity"], "confidence": 0.91}
  ],
  "voice_activity": {"frame_ms": 20, "path": "va/ep012.npz"},
  "words": [
    {"id": "w812", "spk": "podx_host2", "text": "हाँ", "text_norm": "haan", "lang": "hi",
     "start": 341.22, "end": 341.48, "source": "asr_separated", "confidence": 0.71}
  ],
  "segments": [
    {"id": "ipu_301", "spk": "podx_host1", "start": 338.9, "end": 344.6, "lang": "mixed"}
  ],
  "events": [
    {"id": "ev_88", "type": "backchannel", "attrs": {"function": "agreement"},
     "by": "podx_host2", "over": "podx_host1", "onset": 341.22, "word_ids": ["w812"],
     "n_active": 3, "confidence": 0.74, "rule_version": "v3",
     "judge": {"prompt_version": "completion_v2", "completion": "incomplete"}}
  ]
}
```

**Exporters** (all renderings of the same data):

- Plain transcript, with options to include or drop backchannel words and a rule for ordering overlapping speech
- Tagged transcript (events interleaved)
- VAP format: separated two-channel audio + per-speaker activity lists
- RTTM speaker segments
- Praat TextGrid for visual inspection

**VAP input construction:** VAP is two-party, so build two kinds of windows, used for both evaluation and training:

1. **Two-speaker windows (primary).** Windows where exactly two people speak across the whole context and prediction span. Channels = the two separated streams.
2. **Focal-speaker views (secondary).** For multi-party windows: channel 1 = one speaker, channel 2 = everyone else's separated streams summed; rotate who is focal. This is a compromise, because "everyone else" blends several people's turn-taking, so results are reported separately.

Labels for both come from the Stage 3 voice-activity matrix. Apply the excluded-region and cut padding masks.

### Stage 10 — Gold set, pipeline-loss measurement, and evaluation

#### Gold set

Annotation time is limited, so annotate **sampled windows** rather than whole episodes.

- **Selection:** windows from 6–8 channels (a few episodes each), chosen to span 2 speakers vs 3+, English-heavy vs Hindi-heavy, and low vs high edit density. Over-sample windows with overlaps and speaker changes; long monologues add little. Fetch these episodes with video and keep them locally (`download --video --only gold.txt`).
- **Priority order** (stop when the budget runs out):
  1. Speaker activity in the sampled windows, including short overlaps
  2. Event labels at each onset
  3. Cuts, marked while watching the video
  4. Words inside overlaps
- **Agreement:** two people annotate a small shared portion to measure inter-annotator agreement.
- **Split:** a tuning half and a reporting half.
- **Held out from training:** gold **channels** are excluded entirely from the fine-tuning set (Stage 11), not just the annotated windows or episodes, because hosts recur across a channel's episodes.

**Measured on the gold set:**

| Measurement | Used for |
|---|---|
| Diarizer miss rate by event type and duration | Bias of the event set; second-speaker threshold |
| WER (existing WER framework), per language | ASR quality; script normalization check |
| Cut detector hits and false alarms; timing drift | Cut padding value |
| Separation vs original in `music_bed` regions | Keep-with-separation vs exclude decision |
| Judge accuracy per language | Trust in completion-dependent labels |

#### Pipeline-loss measurement without per-speaker ground truth

There is no per-speaker ground truth for YouTube episodes, so measure the pipeline on proxy data where the truth is known:

- **Mixdown test on a proxy corpus.** Take conversational data with a separate channel or headset mic per speaker (e.g. AMI meetings for multi-party English; any per-speaker-channel Hindi/English conversation data you can access). Mix the channels down, run the full pipeline, and compare against the per-channel truth. This measures diarization, separation, and event-detection loss on similar, though not identical, audio.
- **Synthetic edit test.** On the same proxy data, apply YouTube-style edits: shorten pauses, add jump cuts, trim crosstalk, add ducked music. This measures how well Stages 1–2 find the edits, and how much edited timing shifts VAP metrics.
- **Gold-set timing.** For a sample of overlaps in the gold windows, mark onsets precisely by hand. This is the only timing check on the target audio; keep it small but careful.

Treat proxy results as estimates: language, room acoustics, and editing style differ from your episodes.

#### VAP evaluation protocol

- **Metrics:** the standard VAP metrics from voice activity (shift vs hold at pauses, shift prediction, backchannel prediction), plus analysis of model predictions at onsets of each custom event class.
- **Splits:** by channel.
- **Report each result stratified by:**
  - confidence tier
  - condition: clean / `music_bed` / near cut / overlap-dependent
  - speaker setup: two-speaker windows vs focal-speaker views
  - language: `en` / `hi` / `mixed` / code-switch points. The original VAP models were trained on English telephone conversation, so this doubles as a transfer test.

### Stage 11 — Fine-tuning

**What training needs:** VAP's training target is derived from voice activity alone, so the fine-tuning set is the separated two-channel audio (Stage 4) plus the voice-activity matrix (Stage 3). Event labels are not part of the loss; they are used to check whether fine-tuning helped on the events you care about.

**Building the training set:**

- Start from a pretrained VAP checkpoint; don't train from scratch on this amount of data.
- Use only channels outside the gold set.
- Read audio and annotations from the `chunks` subset on HF; local copies are deleted after upload. Chunk boundaries are cut points: VAP windows must not cross them.
- Drop excluded regions and windows inside the cut padding.
- Keep high-confidence windows; treat the confidence threshold as an experiment variable rather than a fixed value.
- Include `music_bed` windows only if separation passed the gold-set test in Stage 10.
- Start with two-speaker windows; add focal-speaker views only if an ablation shows they help.

**Risks specific to training on pipeline output:**

| Risk | Why it matters | Mitigation |
|---|---|---|
| The model learns the diarizer's mistakes | Missed short backchannels in the targets teach the model to under-predict backchannels | Filter or down-weight low-confidence windows; check backchannel metrics on gold specifically; optionally derive soft targets from diarizer probabilities |
| The model learns separation artifacts | At deployment it may see real per-speaker channels or a different separator | Also evaluate on proxy data with real channels; mix in clean two-channel conversational data |
| The model learns edited pacing | Even with cut padding, edited episodes have shorter pauses than natural talk | Compare your pause and gap distributions with natural conversation data; mix in natural data if they differ a lot |
| Same hosts in train and test | Results reflect these speakers, not people in general | Split by channel; hold out whole channels as the test set |

**Experiments:**

- Pretrained baseline vs fine-tuned, on the gold reporting half, with the full Stage 10 stratification (confidence, condition, speaker setup, language).
- Ablations: confidence threshold, with/without focal-speaker views, with/without `music_bed` windows, with/without mixed-in natural data.
- Track runs in MLflow, tying each model to the pipeline and config version that produced its training data.

---

## 6. Starting configuration values

All values are starting points to tune on the gold tuning half.

| Parameter | Start value |
|---|---|
| VAP frame size | 20 ms |
| IPU merge gap | ~200 ms |
| Backchannel max duration | ~1 s, word-list tokens only |
| A yielded | A stops within ~300–500 ms of B onset and doesn't resume within ~1 s |
| B took floor | B continues ≥ ~1 s or beyond word-list tokens |
| Competitive interruption | Both active > ~500 ms after B onset |
| Barge-in | A stops within ~200–300 ms of B onset, cut off mid-word or mid-phrase |
| Cut padding | A few seconds before; ≥ prediction horizon (~2 s) after |
| Music region min duration | ~1 s |
| ASR chunk length | ≤ 55 min (cut searched in the last 2 min) |
| Batch size | ~10 h of audio (~2 GB peak local storage: chunks + both parquet files) |
| Episode duration filter | 15 min – 4 h |

## 7. Build order and milestones

| # | Milestone | Done when |
|---|---|---|
| 0 | Acquisition + batch loop on 2–3 channels | Both subsets on the private HF dataset; Chat reports arrive; local work dir empty after each batch |
| 1 | Repeated-asset discovery per channel | Intro/outro/sting templates extracted; matches located in every episode |
| 2 | One two-speaker episode end to end (simplest version of every stage) | All layers produced; inspected in Praat; exports work |
| 3 | Gold set annotated | Sampled windows from 3–4 episodes annotated in priority order; agreement measured |
| 4 | Fix the weakest stage per gold measurements | Likely backchannel recall in overlaps; second-speaker threshold tuned |
| 5 | Multi-party handling | Floor tracking, `by`/`over` attribution, focal-speaker VAP views |
| 6 | Proxy-corpus mixdown and synthetic edit tests | Pipeline loss and edit-detection quality quantified |
| 7 | Scale to all episodes; baseline VAP evaluation | Stratified results report for the pretrained model |
| 8 | Fine-tuning | Fine-tuned vs baseline on the gold reporting half, with ablations |

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Diarizer misses short overlapped backchannels | Event set biased toward louder, longer backchannels | Second-speaker threshold, overlap ASR pass, measured miss rate, confidence tiers |
| ~80 ms diarizer timing precision | Onset timing noisier than VAP's 20 ms frames | VAD refinement in single-speaker regions; report precision limits |
| Missed cuts | Silently corrupted timing | Over-detect; padding; hand-marked cuts in gold set |
| Separation artifacts thin out quiet speech | Lost backchannels and laughter | Gold-set comparison before choosing separation |
| Code-switching breaks ASR or script consistency | Word-list misses, noisy WER | Fixed script convention, normalization, pilot on gold minutes |
| English-centric sense of sentence completion | Interruption labels wrong in Hindi stretches | Hinglish examples for judge; per-language judge accuracy |
| Multi-party talk doesn't fit two-party VAP | Hard-to-interpret results | Separate two-speaker and focal-speaker evaluation sets |
| No timing ground truth for the target audio | Pipeline loss on these episodes can only be estimated | Proxy-corpus mixdown and synthetic edit tests; small, precise gold timing sample |
| Fine-tuning on pipeline output | Model inherits diarizer, separation, and editing artifacts | Confidence filtering, gold-only evaluation, mixed-in natural data (Stage 11) |
| YouTube blocks downloads or yt-dlp breaks | Runs stall | Sequential downloads with pauses, cookies file, stop after 5 failures in a row with a Chat alert, update yt-dlp often |
| Episodes are split into ≤ 55 min chunks | Speaker numbers restart per chunk; a turn can be split at a boundary | Cuts at the quietest point or inside excluded regions; diarizer links speakers across chunks; no VAP window crosses a boundary |
| HF org storage quota | Uploads fail once the org's private storage is full (~120 MB per audio hour) | Upload failure stops the run with a Chat alert and keeps local files; check the quota before scaling up |
| Re-uploads, clips and compilations | Duplicate speech inflates and leaks across splits | Title filter and duration bounds now; audio-fingerprint dedupe in Stage 1 |
| Auto-dubbed audio tracks | Translated speech instead of the original conversation | Chosen audio format (language, note) recorded per episode; check on the gold set |
| Sparse SponsorBlock labels for Hindi content | Sponsor reads and intros slip through | Per-channel repeated-asset discovery and the music classifier (Stage 1) |
| Very uneven channel sizes | A few channels dominate training and metrics | Cap episodes per source (`max_episodes`); report per channel |

## 9. Open decisions

- **Hindi script convention:** Devanagari, romanized, or mixed. Pick whatever downstream models consume; everything is normalized for matching anyway.
- **Channel list:** which channels and how many episodes each (`sources.toml`).
- **HF org and storage:** which org holds the dataset, and whether its private storage quota covers the planned hours.