import json

import pytest

from cactus import _telemetry
from cactus.agent import fetch


@pytest.fixture
def sent(monkeypatch):
    requests = []
    monkeypatch.setattr(_telemetry, "_get_anon_id", lambda: "test")
    monkeypatch.setattr(_telemetry.urllib.request, "urlopen",
                        lambda request, timeout: requests.append(request))
    return requests


@pytest.mark.parametrize(("event", "props"), [
    ("complete", {"n_tools": 1, "tuned": False, "generation": 3}),
    ("transcribe", {"tuned": False, "timestamps": False}),
    ("stream", {"tuned": False}),
    ("cli:download", {}),
])
def test_every_tracked_event_is_sent(sent, event, props):
    _telemetry._send(event, props)

    assert len(sent) == 1
    payload = json.loads(sent[0].data)
    assert payload["event"] == event
    assert payload["engine"] == fetch.ENGINE_VERSION
    assert payload["props"] == props
