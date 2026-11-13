import tomllib

from youtube_transcribe.find import FindConfig, rank_channels, render_sources, slug
from youtube_transcribe.youtube import Filters

FILTERS = Filters(min_duration_s=1800, max_duration_s=14400, title_exclude=r"(?i)\bshorts?\b")
CONFIG = FindConfig(queries=["q1", "q2"], min_hits=2, max_channels=5, max_episodes=7)


def hit(video: str, channel: str | None, channel_id: str | None, title: str = "Ep", minutes: int = 60) -> dict:
    return {"id": video, "title": title, "channel": channel, "channel_id": channel_id, "duration": minutes * 60}


RESULTS = {
    "q1": [
        hit("a1", "Good Pod", "UC_good", "पॉडकास्ट एक"),
        hit("a2", "Good Pod", "UC_good", "पॉडकास्ट दो"),
        hit("b1", "One Off", "UC_oneoff"),  # a single hit: not a series
        hit("c1", "Clips", "UC_clips", minutes=3),  # too short
        hit("d1", None, None),  # collab video: no channel id
        hit("e1", "Short Pod", "UC_short", "Best shorts"),  # title excluded
    ],
    "q2": [
        hit("a3", "Good Pod", "UC_good", "Episode three"),
        hit("f1", "Other Pod", "UC_other"),
        hit("f2", "Other Pod", "UC_other"),
    ],
}


def test_rank_channels_keeps_series_and_orders_by_queries_then_hits():
    channels = rank_channels(RESULTS, FILTERS, CONFIG)
    assert [(c.id, c.hits, len(c.queries)) for c in channels] == [("UC_good", 3, 2), ("UC_other", 2, 1)]


def test_max_channels_caps_the_list():
    capped = FindConfig(queries=["q1", "q2"], min_hits=1, max_channels=1)
    assert [c.id for c in rank_channels(RESULTS, FILTERS, capped)] == ["UC_good"]


def test_rendered_sources_file_is_valid_toml_for_discover():
    channels = rank_channels(RESULTS, FILTERS, CONFIG)
    parsed = tomllib.loads(render_sources(channels, FILTERS, CONFIG))
    assert Filters(**parsed["filters"]) == FILTERS
    good, other = parsed["sources"]
    assert good == {
        "name": "good-pod",
        "url": "https://www.youtube.com/channel/UC_good/videos",
        "lang": "hi",  # 2 of 3 titles are Devanagari
        "max_episodes": 7,
    }
    assert other["lang"] == "hi-en"


def test_slug_falls_back_to_channel_id_for_non_latin_names():
    assert slug("पुलियाबाज़ी", "UChrTy") == "uchrty"
    assert slug("Raj Shamani!", "x") == "raj-shamani"
