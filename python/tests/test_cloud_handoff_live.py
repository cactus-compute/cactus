from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from .bundles import _find_bundle

PREFERRED_BUNDLES = ["qwen3-0.6b", "lfm2-350m"]
OPTIONS = {"max_tokens": 8, "temperature": 0, "confidence_threshold": 1.0}
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


class _AuthFailHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        with self.server.hits_lock:
            self.server.hits += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"error": "invalid or expired api key"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _hits(server) -> int:
    with server.hits_lock:
        return server.hits


def test_auth_failure_disables_cloud_handoff_for_session(monkeypatch, tmp_path):
    bundle = _find_bundle(PREFERRED_BUNDLES, {"qwen", "lfm2"}, on_missing=pytest.skip)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthFailHandler)
    server.hits = 0
    server.hits_lock = threading.Lock()
    threading.Thread(target=server.serve_forever, daemon=True).start()

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CACTUS_CLOUD_API_BASE", f"http://127.0.0.1:{server.server_address[1]}/api/v1")
    monkeypatch.setenv("CACTUS_CLOUD_KEY", "stale-test-key")
    monkeypatch.setenv("CACTUS_NO_CLOUD_TELE", "1")
    monkeypatch.delenv("CACTUS_DISABLE_CLOUD_HANDOFF", raising=False)

    from cactus.bindings.cactus import cactus_complete, cactus_destroy, cactus_init

    model = cactus_init(str(bundle))
    try:
        with pytest.raises(RuntimeError):
            cactus_complete(model, MESSAGES, options=OPTIONS)
        assert _hits(server) == 1

        start = time.monotonic()
        try:
            result = cactus_complete(model, MESSAGES, options=OPTIONS)
        except RuntimeError:
            result = None
        elapsed = time.monotonic() - start

        assert _hits(server) == 1
        assert result is not None
        assert result["success"] is True
        assert elapsed < 5.0
    finally:
        cactus_destroy(model)
        server.shutdown()
        server.server_close()
