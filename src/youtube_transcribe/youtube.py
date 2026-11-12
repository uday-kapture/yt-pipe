"""Find podcast episodes on YouTube and fetch them: 16 kHz audio, metadata, SponsorBlock regions, ASR chunks."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import soundfile
import yt_dlp

from .audio import MAX_CHUNK_S, frame_rms, plan_chunks, to_16k_mono, write_chunks

log = logging.getLogger(__name__)

LIVE_STATUSES = frozenset({"is_live", "is_upcoming", "post_live"})
# SponsorBlock categories that are not conversation. "filler" (tangents) and "hook" (greetings) are still talk.
EXCLUDE_CATEGORIES = ["sponsor", "selfpromo", "interaction", "intro", "outro", "preview", "music_offtopic"]
YOUTUBE_FIELDS = ["id", "title", "description", "channel", "channel_id", "upload_date", "duration", "license",
                  "language", "webpage_url", "chapters"]  # fmt: skip
FORMAT_FIELDS = ["format_id", "format_note", "language", "acodec", "abr"]
SLEEP_S = 5  # pause before each download; keeps a long run under YouTube's rate limits


@dataclass
class Filters:
    min_duration_s: float = 900
    max_duration_s: float = 14400
    title_exclude: str = ""


def filter_reason(entry: dict, filters: Filters) -> str | None:
    """Why a flat playlist entry is skipped, or None to keep it."""
    if entry.get("live_status") in LIVE_STATUSES:
        return "live"
    duration = entry.get("duration")
    if not duration:
        return "no_duration"
    if duration < filters.min_duration_s:
        return "too_short"
    if duration > filters.max_duration_s:
        return "too_long"
    if filters.title_exclude and re.search(filters.title_exclude, entry.get("title") or ""):
        return "title"
    return None


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _videos(ydl: yt_dlp.YoutubeDL, info: dict):
    """Flat video entries, opening nested playlists (channel root, /podcasts tab) as they come."""
    for entry in info.get("entries") or []:
        if not entry:
            continue
        if entry.get("entries") is not None:
            yield from _videos(ydl, entry)
        elif entry.get("ie_key") == "YoutubeTab":
            yield from _videos(ydl, ydl.extract_info(entry["url"], download=False))
        else:
            yield entry


def discover(sources: list[dict], filters: Filters, out_dir: Path) -> list[dict]:
    """Expand sources into episodes that pass the filters; writes ``catalog.jsonl``."""
    rows: list[dict] = []
    seen: set[str] = set()
    with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True}) as ydl:
        for source in sources:
            limit = source.get("max_episodes")
            kept, skips = [], Counter()
            try:
                for entry in _videos(ydl, ydl.extract_info(source["url"], download=False)):
                    if limit is not None and len(kept) >= limit:
                        break
                    reason = filter_reason(entry, filters) or ("duplicate" if entry["id"] in seen else None)
                    if reason:
                        skips[reason] += 1
                        continue
                    seen.add(entry["id"])
                    kept.append(
                        {
                            "episode_id": f"yt_{entry['id']}",
                            "url": f"https://www.youtube.com/watch?v={entry['id']}",
                            "title": entry.get("title"),
                            "duration": entry["duration"],
                            "source": source["name"],
                            "lang_hint": source.get("lang"),
                        }
                    )
            except yt_dlp.utils.DownloadError as error:  # one bad source must not stop the rest
                log.error("source %s failed: %s", source["name"], error)
            hours = sum(r["duration"] for r in kept) / 3600
            log.info("%s: kept %d episodes (%.1f h), skipped %s", source["name"], len(kept), hours, dict(skips))
            rows += kept
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "catalog.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    log.info("catalog: %d episodes, %.1f h", len(rows), sum(r["duration"] for r in rows) / 3600)
    return rows


def _download(url: str, outtmpl: Path, cookies_file: str | None, **opts) -> dict:
    opts |= {"outtmpl": str(outtmpl), "quiet": True, "noprogress": True, "sleep_interval": SLEEP_S}
    if cookies_file:
        opts["cookiefile"] = cookies_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


def _video(row: dict, ep_dir: Path, cookies_file: str | None) -> str:
    info = _download(row["url"], ep_dir / "video_360p.%(ext)s", cookies_file, format="bv[height<=360]/wv")
    return Path(info["requested_downloads"][0]["filepath"]).name


def fetch_episode(
    row: dict, ep_root: Path, asr_root: Path, cookies_file: str | None = None, video: bool = False
) -> dict:
    """Fetch one episode into ``ep_root/<episode_id>/`` and its ASR chunks into ``asr_root/<episode_id>/``.

    ``episode.json`` is written last, so its presence means the episode is complete.
    Paths in it are relative to ``ep_root.parent``.
    """
    episode_id = row["episode_id"]
    ep_dir = ep_root / episode_id
    done = ep_dir / "episode.json"
    if done.exists():
        episode = json.loads(done.read_text())
        if video and not episode["video"]:
            episode["video"] = _video(row, ep_dir, cookies_file)
            write_json(done, episode)
        return episode

    ep_dir.mkdir(parents=True, exist_ok=True)
    info = _download(
        row["url"],
        ep_dir / "audio_orig.%(ext)s",
        cookies_file,
        format="bestaudio/best",
        postprocessors=[{"key": "SponsorBlock", "categories": EXCLUDE_CATEGORIES, "when": "after_filter"}],
    )
    downloaded = info["requested_downloads"][0]
    audio_16k = ep_dir / "audio_16k.flac"
    to_16k_mono(Path(downloaded["filepath"]), audio_16k)
    duration = soundfile.info(str(audio_16k)).duration

    excluded = [
        {"start": c["start_time"], "end": c["end_time"], "reason": c["category"], "method": "sponsorblock",
         "confidence": None, "action": "exclude"}
        for c in info.get("sponsorblock_chapters") or []
    ]  # fmt: skip
    rms = frame_rms(audio_16k) if duration > MAX_CHUNK_S else None
    spans = plan_chunks(rms, duration, [(r["start"], r["end"]) for r in excluded])
    chunk_paths = write_chunks(audio_16k, spans, asr_root / episode_id)
    base = ep_root.parent

    episode = {
        "episode_id": episode_id,
        "versions": {"pipeline": version("youtube-transcribe"), "yt_dlp": yt_dlp.version.__version__},
        "source": {"name": row["source"], "url": row["url"], "lang_hint": row.get("lang_hint")},
        "youtube": {k: info.get(k) for k in YOUTUBE_FIELDS},
        "audio": {
            "original": Path(downloaded["filepath"]).name,
            "path_16k": str(audio_16k.relative_to(base)),
            "duration_s": round(duration, 3),
            # which track was picked; guards against auto-dubbed audio
            "format": {k: downloaded.get(k) for k in FORMAT_FIELDS},
        },
        "video": _video(row, ep_dir, cookies_file) if video else None,
        "excluded_regions": excluded,
        # ids match nemo_3_5_transcribe.client.conversation_id() run on asr_root
        "asr_chunks": [
            {"id": f"{episode_id}__{path.stem}", "path": str(path.relative_to(base)), "start": start, "end": end}
            for (start, end), path in zip(spans, chunk_paths, strict=True)
        ],
    }
    write_json(done, episode)
    return episode
