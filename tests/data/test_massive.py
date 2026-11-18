"""Massive client: transient failures are retried, others raise at once."""

import pytest
import requests

from gnns.data import massive


class Resp:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def _client(monkeypatch, statuses):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        s = statuses[len(calls)]
        calls.append(s)
        if s == "drop":
            raise requests.ConnectionError("dropped")
        return Resp(s)

    monkeypatch.setattr(massive.requests, "get", get)
    monkeypatch.setattr(massive.time_mod, "sleep", lambda s: None)
    return calls


def test_rate_limits_server_errors_and_dropped_connections_are_retried(monkeypatch):
    calls = _client(monkeypatch, [429, 503, "drop", 200])
    assert massive._request("u", {}, {}).status_code == 200
    assert calls == [429, 503, "drop", 200]


def test_a_client_error_raises_at_once(monkeypatch):
    calls = _client(monkeypatch, [404, 200])
    with pytest.raises(requests.HTTPError):
        massive._request("u", {}, {})
    assert calls == [404]


def test_retries_give_up_after_the_last_wait(monkeypatch):
    calls = _client(monkeypatch, [503] * 6)
    with pytest.raises(requests.HTTPError):
        massive._request("u", {}, {})
    assert len(calls) == len(massive.RETRY_WAITS) + 1
