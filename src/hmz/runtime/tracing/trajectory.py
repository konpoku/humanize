"""Complete emitted trajectories, recorded by the trusted run owner."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from hmz.coganchor.agents import AgentBase, Event, SessionBase

__all__ = ["Trajectory", "TrajectoryError"]


class TrajectoryError(RuntimeError):
    """The run cannot provide a complete, durable emitted trajectory."""


class Trajectory:
    """A new durable JSONL journal outside the optimizer's writable filesystem.

    Codex items are retained without display truncation at both ends of each call.
    Native subagents are refused because their child tool streams are not completely
    observable. This records emitted evidence, never unreported model reasoning.

    Args:
      journal: A new file in the trusted launcher's filesystem.
    """

    def __init__(self, journal: str | os.PathLike[str]) -> None:
        self._lock = threading.RLock()
        self._sequence = 0
        self._previous = "0" * 64
        self._turns: dict[tuple[str, str], int] = {}
        self._sessions: dict[int, tuple[SessionBase, str]] = {}
        self._pending: set[str] = set()
        self._completed: set[str] = set()
        self._failure: TrajectoryError | None = None
        self._closed = False
        fd = os.open(Path(journal), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        try:
            self._write("opened", {"schema": "hmz.trajectory.v1"})
        except BaseException:
            self._file.close()
            raise

    @property
    def failure(self) -> TrajectoryError | None:
        """The first evidence failure, which later events cannot undo."""
        return self._failure

    def opened(self, agent: AgentBase) -> None:
        """Refuse unsupported backends before their first turn."""
        with self._lock:
            self._check()
            if agent.backend != "codex":
                self._refuse("unsupported backend: complete tool evidence is required")

    def event(
        self, agent: AgentBase, session: SessionBase | None, event: Event
    ) -> None:
        """Persist an event, failing closed when complete tool evidence is missing."""
        with self._lock:
            self._check()
            try:
                self._event(agent, session, event)
            except TrajectoryError:
                raise
            except Exception as error:  # noqa: BLE001 -- evidence failures are fatal
                self._refuse(f"trajectory evidence failed: {type(error).__name__}")

    def _event(
        self, agent: AgentBase, session: SessionBase | None, event: Event
    ) -> None:
        self.opened(agent)
        if session is None:
            session_id = ""
        else:
            if id(session) not in self._sessions:
                self._sessions[id(session)] = (
                    session,
                    f"session-{len(self._sessions) + 1}",
                )
            session_id = self._sessions[id(session)][1]
        owner = (str(agent.id), session_id)
        if event.kind == "begins":
            self._turns[owner] = self._turns.get(owner, 0) + 1
        turn = self._turns.get(owner, 0)
        self._write(
            "event",
            {
                "agent": owner[0],
                "session": owner[1],
                "turn": turn,
                "kind": event.kind,
                "text": event.text,
                "whose": event.whose,
                "tokens": dict(event.tokens),
                "spent": dict(event.spent),
                "tool": event.tool,
            },
        )
        if event.kind not in {"tool", "tool-result", "subagent", "subagent-ends"}:
            return
        tool = event.tool
        if (
            tool is None
            or not isinstance(tool.get("name"), str)
            or not tool["name"]
            or not isinstance(tool.get("id"), str)
            or not tool["id"]
            or tool.get("phase") not in {"started", "completed"}
            or not isinstance(tool.get("native"), dict)
        ):
            self._refuse("missing structured tool evidence")
        if event.kind in {"subagent", "subagent-ends"}:
            self._refuse("native subagents have incomplete tool coverage")
        call_id = json.dumps([*owner, turn, tool["id"]], ensure_ascii=False)
        if tool["phase"] == "started":
            if call_id in self._pending or call_id in self._completed:
                self._refuse("duplicate tool start")
            self._pending.add(call_id)
        else:
            if call_id in self._completed:
                self._refuse("duplicate tool completion")
            # Some Codex items are emitted only as completed. Their full payload is
            # retained and inspected even when no start notification was available.
            self._pending.discard(call_id)
            self._completed.add(call_id)

    def finish(self, *, aborted: bool = False) -> None:
        """Seal the journal; successful runs must have no unfinished tool calls."""
        with self._lock:
            if self._closed:
                if self._failure is not None:
                    raise self._failure
                return
            try:
                if self._failure is None and self._pending and not aborted:
                    self._refuse("tool completions are missing")
                self._write(
                    "finished",
                    {
                        "status": "failed"
                        if self._failure
                        else "aborted"
                        if aborted
                        else "complete",
                        "pending": sorted(self._pending),
                        "completed": len(self._completed),
                    },
                )
            except TrajectoryError:
                raise
            except Exception as error:
                self._failure = self._failure or TrajectoryError(
                    f"trajectory sealing failed: {type(error).__name__}"
                )
                raise self._failure from error
            finally:
                self._closed = True
                self._file.close()
            if self._failure is not None:
                raise self._failure

    def _check(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise TrajectoryError("the trajectory is closed")

    def _refuse(self, reason: str) -> NoReturn:
        if self._failure is None:
            self._failure = TrajectoryError(reason)
            with contextlib.suppress(Exception):
                self._write("failure", {"reason": reason})
        raise self._failure

    def _write(self, kind: str, payload: Mapping[str, Any]) -> None:
        self._sequence += 1
        record = {
            "sequence": self._sequence,
            "time_ns": time.time_ns(),
            "type": kind,
            "previous_sha256": self._previous,
            **payload,
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        self._previous = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self._file.write(encoded + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
