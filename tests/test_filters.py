import pytest

from youtube_transcribe.youtube import Filters, filter_reason

FILTERS = Filters(min_duration_s=900, max_duration_s=14400, title_exclude=r"(?i)#shorts|\btrailer\b")


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param({"duration": 3600, "title": "Episode 12 ft. a guest"}, None, id="keep"),
        pytest.param({"duration": 300, "title": "Episode 12"}, "too_short", id="too-short"),
        pytest.param({"duration": 20000, "title": "Episode 12"}, "too_long", id="too-long"),
        pytest.param({"duration": 3600, "live_status": "is_upcoming"}, "live", id="upcoming-live"),
        pytest.param({"duration": None, "title": "Episode 12"}, "no_duration", id="no-duration"),
        pytest.param({"duration": 3600, "title": "Season 2 Trailer"}, "title", id="title-excluded"),
    ],
)
def test_filter_reason(entry, expected):
    assert filter_reason(entry, FILTERS) == expected
