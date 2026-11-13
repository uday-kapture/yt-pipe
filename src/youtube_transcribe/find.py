"""Find podcast channels by searching YouTube, and write them as a ``sources`` file.

Search returns videos, so hits are filtered like ``discover`` filters episodes, then grouped by channel:
a channel that shows up with several long-form episodes (and across several queries) looks like a series.
The output is in ``sources.toml`` format. Review and edit it, or pass it straight to ``discover``.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp

from .youtube import Filters, filter_reason

log = logging.getLogger(__name__)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass
class FindConfig:
    queries: list[str]
    results_per_query: int = 50
    min_hits: int = 2  # a channel needs this many passing episodes across all queries
    max_channels: int = 40
    max_episodes: int = 20  # written into each source
    default_lang: str = "hi-en"


@dataclass
class Channel:
    id: str
    names: list[str] = field(default_factory=list)
    queries: set[str] = field(default_factory=set)
    episodes: dict[str, dict] = field(default_factory=dict)  # video id -> entry

    @property
    def name(self) -> str:
        return max(set(self.names), key=self.names.count).strip()

    @property
    def hits(self) -> int:
        return len(self.episodes)

    @property
    def median_minutes(self) -> float:
        return statistics.median(e["duration"] for e in self.episodes.values()) / 60

    @property
    def devanagari_share(self) -> float:
        titles = [e.get("title") or "" for e in self.episodes.values()]
        return sum(bool(_DEVANAGARI.search(t)) for t in titles) / len(titles)


def search(queries: list[str], results_per_query: int) -> dict[str, list[dict]]:
    """Flat YouTube search results per query."""
    results = {}
    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True, "ignoreerrors": True}) as ydl:
        for query in queries:
            info = ydl.extract_info(f"ytsearch{results_per_query}:{query}", download=False) or {}
            results[query] = [e for e in info.get("entries") or [] if e]
            log.info("%r: %d results", query, len(results[query]))
    return results


def rank_channels(results: dict[str, list[dict]], filters: Filters, config: FindConfig) -> list[Channel]:
    """Group passing hits by channel; keep series-like channels, best first."""
    channels: dict[str, Channel] = {}
    for query, entries in results.items():
        for entry in entries:
            if not entry.get("channel_id") or filter_reason(entry, filters):  # collabs have no channel id
                continue
            channel = channels.setdefault(entry["channel_id"], Channel(entry["channel_id"]))
            channel.names.append(entry.get("channel") or entry["channel_id"])
            channel.queries.add(query)
            channel.episodes[entry["id"]] = entry
    kept = [c for c in channels.values() if c.hits >= config.min_hits]
    kept.sort(key=lambda c: (len(c.queries), c.hits), reverse=True)
    return kept[: config.max_channels]


def slug(name: str, fallback: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or fallback.lower()


def render_sources(channels: list[Channel], filters: Filters, config: FindConfig) -> str:
    """A ``sources.toml`` with one commented block per channel."""
    lines = [
        "# Written by `youtube-transcribe find`. Delete the channels you don't want, then use it as --config.",
        "",
        "[filters]",
        f"min_duration_s = {filters.min_duration_s:g}",
        f"max_duration_s = {filters.max_duration_s:g}",
        f"title_exclude = {json.dumps(filters.title_exclude)}",
    ]
    used: set[str] = set()
    for c in channels:
        name = slug(c.name, c.id)
        name = name if name not in used else f"{name}-{c.id[-4:].lower()}"
        used.add(name)
        lang = "hi" if c.devanagari_share >= 0.3 else config.default_lang
        example = next(iter(c.episodes.values()))["title"]
        lines += [
            "",
            f"# {c.name}: {c.hits} episodes in {len(c.queries)} queries, median {c.median_minutes:.0f} min, "
            f"{c.devanagari_share:.0%} Devanagari titles",
            f"# e.g. {example[:90]!r}".replace("\n", " "),
            "[[sources]]",
            f"name = {json.dumps(name)}",
            f'url = "https://www.youtube.com/channel/{c.id}/videos"',
            f'lang = "{lang}"',
            f"max_episodes = {config.max_episodes}",
        ]
    return "\n".join(lines) + "\n"


def find(config: FindConfig, filters: Filters, out: Path) -> list[Channel]:
    channels = rank_channels(search(config.queries, config.results_per_query), filters, config)
    out.write_text(render_sources(channels, filters, config))
    hours = sum(c.median_minutes * config.max_episodes for c in channels) / 60
    log.info(
        "%d channels written to %s (up to ~%.0f h at %d episodes each)", len(channels), out, hours, config.max_episodes
    )
    return channels
