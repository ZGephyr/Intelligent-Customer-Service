"""Redis 历史缓存测试。"""

from __future__ import annotations

from rag_service.core.types import ChatTurn
from rag_service.infra.redis_store import HistoryStore


class FakeRedis:
    def __init__(self):
        self.data: dict[str, list[str]] = {}
        self.expire_calls = []

    def rpush(self, key, *values):
        self.data.setdefault(key, []).extend(values)

    def ltrim(self, key, start, end):
        data = self.data[key]
        if end == -1:
            self.data[key] = data[start:]
        else:
            self.data[key] = data[start : end + 1]

    def lrange(self, key, start, end):
        data = self.data.get(key, [])
        if end == -1:
            return data[start:]
        return data[start : end + 1]

    def expire(self, key, ttl):
        self.expire_calls.append((key, ttl))

    def ping(self):
        return True


def test_append_trims_history_to_max_messages():
    store = HistoryStore(FakeRedis(), ttl_seconds=60, max_messages=2)
    store.append(
        "s1",
        [ChatTurn("user", "a"), ChatTurn("assistant", "b"), ChatTurn("user", "c")],
    )
    assert [turn.content for turn in store.get("s1")] == ["b", "c"]


def test_append_sets_ttl_on_session_key():
    client = FakeRedis()
    HistoryStore(client, ttl_seconds=60, max_messages=20).append("s1", [ChatTurn("user", "a")])
    assert client.expire_calls == [("chat:history:s1", 60)]


def test_get_when_redis_raises_returns_empty_list():
    class Broken(FakeRedis):
        def lrange(self, *args):
            raise RuntimeError("down")

    assert HistoryStore(Broken(), 60, 20).get("s1") == []


def test_append_when_redis_raises_does_not_propagate():
    class Broken(FakeRedis):
        def rpush(self, *args):
            raise RuntimeError("down")

    HistoryStore(Broken(), 60, 20).append("s1", [ChatTurn("user", "a")])
