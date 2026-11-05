import ctypes
import json

import pytest

from cactus.agent import mirage, runtime


class Engine:
    def __init__(self):
        self.calls = []

    def cactus_load(self, data, size):
        return 0

    def cactus_models(self):
        return 4

    def cactus_describe(self, rgb, width, height, options, language, keywords, out, capacity):
        self.calls.append((rgb, width, height, json.loads(options), language, keywords))
        out.value = b'{"objects":[],"text":[]}'
        return len(out.value)

    def cactus_last_error(self):
        return b"invalid vision options"


@pytest.fixture
def model(monkeypatch):
    engine = Engine()
    monkeypatch.setattr(runtime, "_lib", lambda: engine)
    monkeypatch.setattr(mirage, "_loaded", None)
    return mirage.Mirage(weights=__file__), engine


def test_rgb_options_and_language_reach_the_native_api(model):
    vision, engine = model
    pixels = bytes(range(18))
    assert vision.describe(pixels, width=3, height=2, language="pl", keywords=["Łódź", "café"],
                           options={"sections": ["objects"]}, mask_format="bitmap") == {"objects": [], "text": []}
    assert engine.calls == [(pixels, 3, 2, {"sections": ["objects"], "mask_format": "bitmap"},
                             b"pl", "Łódź\ncafé".encode())]


def test_forms_preserve_field_order_and_constraints(model):
    vision, engine = model
    form = {"animal": {"enum": ["dog", "cat"]}, "legs": {"type": "integer", "minimum": 0, "maximum": 4}}
    vision.extract(b"\0\0\0", form, width=1, height=1)
    assert engine.calls[0][3]["form"] == form
    assert list(engine.calls[0][3]["form"]) == ["animal", "legs"]


@pytest.mark.parametrize("data,width,height", [(b"", 1, 1), (b"123", 0, 1), (b"123", 1, None), (b"123", 1.0, 1)])
def test_invalid_rgb_never_enters_the_engine(model, data, width, height):
    vision, engine = model
    with pytest.raises(ValueError):
        vision.describe(data, width=width, height=height)
    assert not engine.calls


def test_image_arrays_are_rgb_uint8_and_can_be_noncontiguous(model):
    np = pytest.importorskip("numpy")
    vision, engine = model
    pixels = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)[:, ::2]
    vision.describe(pixels)
    assert engine.calls[0][:3] == (pixels.tobytes(), 2, 3)
    for wrong in (pixels.astype("float32"), pixels[:, :, 0]):
        with pytest.raises(ValueError, match="uint8"):
            vision.describe(wrong)


def test_conflicting_options_and_nonfinite_numbers_are_rejected(model):
    vision, engine = model
    with pytest.raises(ValueError, match="both"):
        vision.describe(b"123", width=1, height=1, options={"conf": 0.5}, conf=0.7)
    with pytest.raises(ValueError):
        vision.describe(b"123", width=1, height=1, conf=float("nan"))
    assert not engine.calls


def test_native_errors_are_reported_without_parsing_stale_output(model, monkeypatch):
    vision, engine = model
    vision.describe(b"123", width=1, height=1)
    monkeypatch.setattr(engine, "cactus_describe", lambda *args: -1)
    with pytest.raises(RuntimeError, match="invalid vision options"):
        vision.describe(b"123", width=1, height=1)
