import json

from matensemble.redis.service import RedisService


class _FakeRedis:
    def __init__(self, store):
        self.store = store

    def lrange(self, key, *_args):
        return self.store.get(key, [])

    def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)

    def shutdown(self):
        return None


def test_make_key():
    assert RedisService.make_key("ns", "k") == "ns:k"


def test_register_and_extract_from_stream(monkeypatch):
    service = RedisService(host="h", port=1234)
    backing = {}

    monkeypatch.setattr(
        "redis.Redis",
        lambda **_kwargs: _FakeRedis(backing),
    )

    service.register_on_stream("case1", key="xx", timestep=2, xx=0.2)
    service.register_on_stream("case1", key="xx", timestep=1, xx=0.1)

    df = service.extract_from_stream("case1", key="xx", sort=True)
    assert list(df["timestep"]) == [1, 2]
    assert json.loads(backing["case1:xx"][0])["xx"] == 0.2
