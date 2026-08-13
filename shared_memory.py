
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("shared_memory")


class TemporalIntegrityError(RuntimeError):
    """Raised when a read would leak data from at or after its cutoff."""


def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class SharedMemoryEntry:
    key: str
    value: Any
    written_by: str
    available_at: pd.Timestamp          # when this fact became knowable
    source_available_at: pd.Timestamp | None = None  # provenance, see module docstring
    seq: int = 0                        # assigned by the pool, for stable ordering


@dataclass
class AgentMessage:
    from_agent: str
    to_agent: str
    content: Any
    sent_at: pd.Timestamp
    seq: int = 0


@dataclass
class ToolOutput:
    tool_name: str
    called_by: str
    output: Any
    available_at: pd.Timestamp
    seq: int = 0



class MemoryBackend(Protocol):
    def append(self, bucket: str, record: dict) -> None: ...
    def all(self, bucket: str) -> list[dict]: ...


class InMemoryBackend:

    def __init__(self):
        self._store: dict[str, list[dict]] = {}

    def append(self, bucket: str, record: dict) -> None:
        self._store.setdefault(bucket, []).append(record)

    def all(self, bucket: str) -> list[dict]:
        return list(self._store.get(bucket, []))


class RedisBackend:
    def __init__(self, redis_client, key_prefix: str = "shared_memory"):
        import json  # local import: only needed if this backend is used
        self._json = json
        self.r = redis_client
        self.prefix = key_prefix

    def _k(self, bucket: str) -> str:
        return f"{self.prefix}:{bucket}"

    def append(self, bucket: str, record: dict) -> None:
        self.r.rpush(self._k(bucket), self._json.dumps(record, default=str))

    def all(self, bucket: str) -> list[dict]:
        raw = self.r.lrange(self._k(bucket), 0, -1)
        return [self._json.loads(r) for r in raw]


# ---------------------------------------------------------------------------
# Shared memory pool
# ---------------------------------------------------------------------------

@dataclass
class SharedMemoryPool:


    backend: MemoryBackend = field(default_factory=InMemoryBackend)
    access_log: list[dict] = field(default_factory=list)
    _seq: int = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _log(self, channel: str, action: str, agent: str, as_of: pd.Timestamp | None,
              n_returned: int, n_blocked: int) -> None:
        entry = {"channel": channel, "action": action, "agent": agent,
                  "as_of": as_of, "n_returned": n_returned, "n_blocked": n_blocked}
        self.access_log.append(entry)
        if n_blocked:
            log.warning(
                "temporal integrity: blocked %d %s record(s) for agent=%s at/after as_of=%s (channel=%s)",
                n_blocked, action, agent, as_of, channel,
            )

    # -- shared memory (key/value) ----------------------------------------

    def write(self, agent_id: str, key: str, value: Any, available_at,
              source_available_at=None) -> None:
        entry = SharedMemoryEntry(
            key=key, value=value, written_by=agent_id,
            available_at=_to_utc(available_at),
            source_available_at=_to_utc(source_available_at) if source_available_at is not None else None,
            seq=self._next_seq(),
        )
        record = asdict(entry)
        self.backend.append("memory", record)
        self._log("memory", "write", agent_id, None, 1, 0)

    def read(self, agent_id: str, key: str, as_of, latest_only: bool = True):

        as_of = _to_utc(as_of)
        records = [r for r in self.backend.all("memory") if r["key"] == key]
        before = len(records)
        visible = [r for r in records if _to_utc(r["available_at"]) < as_of]
        blocked = before - len(visible)
        self._log("memory", "read", agent_id, as_of, len(visible), blocked)

        if not visible:
            return None if latest_only else []
        visible.sort(key=lambda r: r["seq"])
        return visible[-1]["value"] if latest_only else [r["value"] for r in visible]

    # -- inter-agent messages ----------------------------------------------

    def send_message(self, from_agent: str, to_agent: str, content: Any, sent_at) -> None:
        msg = AgentMessage(from_agent=from_agent, to_agent=to_agent, content=content,
                            sent_at=_to_utc(sent_at), seq=self._next_seq())
        self.backend.append("messages", asdict(msg))
        self._log("messages", "write", from_agent, None, 1, 0)

    def read_inbox(self, agent_id: str, as_of) -> list[dict]:

        as_of = _to_utc(as_of)
        records = [r for r in self.backend.all("messages") if r["to_agent"] == agent_id]
        before = len(records)
        visible = [r for r in records if _to_utc(r["sent_at"]) < as_of]
        blocked = before - len(visible)
        self._log("messages", "read", agent_id, as_of, len(visible), blocked)
        visible.sort(key=lambda r: r["seq"])
        return visible

    # -- tool outputs --------------------------------------------------------

    def cache_tool_output(self, tool_name: str, called_by: str, output: Any, available_at) -> None:

        rec = ToolOutput(tool_name=tool_name, called_by=called_by, output=output,
                          available_at=_to_utc(available_at), seq=self._next_seq())
        self.backend.append("tool_outputs", asdict(rec))
        self._log("tool_outputs", "write", called_by, None, 1, 0)

    def read_tool_outputs(self, agent_id: str, tool_name: str, as_of) -> list[dict]:
        as_of = _to_utc(as_of)
        records = [r for r in self.backend.all("tool_outputs") if r["tool_name"] == tool_name]
        before = len(records)
        visible = [r for r in records if _to_utc(r["available_at"]) < as_of]
        blocked = before - len(visible)
        self._log("tool_outputs", "read", agent_id, as_of, len(visible), blocked)
        visible.sort(key=lambda r: r["seq"])
        return visible

    # -- provenance check ----------------------------------------------------

    def check_provenance(self, key: str, as_of) -> tuple[bool, int]:
        as_of = _to_utc(as_of)
        records = [r for r in self.backend.all("memory")
                   if r["key"] == key and r.get("source_available_at") is not None]
        violations = sum(
            1 for r in records
            if _to_utc(r["source_available_at"]) >= _to_utc(r["available_at"])
        )
        allowed = violations == 0
        if not allowed:
            log.warning("provenance check failed for key=%s: %d/%d entries claim availability "
                        "earlier than their own source data", key, violations, len(records))
        return allowed, violations

    def save_access_log(self, path) -> None:
        import pathlib
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.access_log).to_parquet(path, index=False)
