from cactus.transpile.audio_preprocess import audio_bucket_frames
from cactus.transpile.audio_preprocess import audio_bucket_seconds


def test_fibonacci_ladder_capped_at_model_window():
    assert audio_bucket_frames(2003) == [100, 200, 300, 500, 800, 1300, 2003]


def test_window_below_ladder_yields_single_bucket():
    assert audio_bucket_frames(80) == [80]


def test_window_equal_to_a_rung_is_not_duplicated():
    assert audio_bucket_frames(800) == [100, 200, 300, 500, 800]


def test_non_positive_window_yields_no_buckets():
    assert audio_bucket_frames(0) == []
    assert audio_bucket_frames(-1) == []


def test_buckets_are_sorted_and_unique():
    buckets = audio_bucket_frames(2003)
    assert buckets == sorted(buckets)
    assert len(buckets) == len(set(buckets))


def test_default_ladder_is_fibonacci_seconds():
    assert audio_bucket_seconds() == (1, 2, 3, 5, 8, 13)


def test_env_override_replaces_ladder(monkeypatch):
    monkeypatch.setenv("CACTUS_TRANSPILER_AUDIO_BUCKETS", "2,4,6")
    assert audio_bucket_seconds() == (2, 4, 6)
    assert audio_bucket_frames(2003) == [200, 400, 600, 2003]


def test_env_override_is_sorted_and_deduplicated(monkeypatch):
    monkeypatch.setenv("CACTUS_TRANSPILER_AUDIO_BUCKETS", "6, 2, 4, 2")
    assert audio_bucket_seconds() == (2, 4, 6)


def test_malformed_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CACTUS_TRANSPILER_AUDIO_BUCKETS", "not-a-number")
    assert audio_bucket_seconds() == (1, 2, 3, 5, 8, 13)


def test_empty_env_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CACTUS_TRANSPILER_AUDIO_BUCKETS", "0,-3")
    assert audio_bucket_seconds() == (1, 2, 3, 5, 8, 13)
