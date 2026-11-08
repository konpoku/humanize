"""Complete emitted evidence is retained or the run fails closed."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from hmz.runtime.trajectory import Trajectory, TrajectoryError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

AGENT = SimpleNamespace(id="worker", backend="codex")
SESSION = SimpleNamespace()


@dataclass
class Evidence:
    kind: str
    text: str = ""
    tool: Mapping[str, Any] | None = None
    whose: str = ""
    tokens: dict[str, int] = field(default_factory=dict[str, int])
    spent: dict[str, float] = field(default_factory=dict[str, float])


def tool(
    phase: str, *, name: str = "commandExecution", call_id: str = "one"
) -> Evidence:
    item = {
        "type": name,
        "id": call_id,
        "command": "python benchmark.py --size 1000",
        "aggregatedOutput": "all output\n" * 200,
    }
    return Evidence(
        kind="tool" if phase == "started" else "tool-result",
        text="a short display label",
        tool={
            "name": name,
            "id": call_id,
            "phase": phase,
            "native": {"method": f"item/{phase}", "params": {"item": item}},
        },
    )


def emit(trajectory: Trajectory, event: Evidence) -> None:
    trajectory.event(cast("Any", AGENT), cast("Any", SESSION), cast("Any", event))


def test_complete_tool_inputs_outputs_and_prompt_are_durable(tmp_path: Path) -> None:
    journal = tmp_path / "trajectory.jsonl"
    trajectory = Trajectory(journal)
    emit(trajectory, Evidence("begins", "the fixed condition prompt"))
    emit(trajectory, tool("started"))
    emit(trajectory, tool("completed"))
    emit(trajectory, Evidence("result", "finished"))
    trajectory.finish()

    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[1]["text"] == "the fixed condition prompt"
    assert rows[-1]["status"] == "complete"
    assert rows[-1]["completed"] == 1
    assert (
        rows[2]["tool"]["native"]["params"]["item"]["aggregatedOutput"]
        == "all output\n" * 200
    )
    previous = "0" * 64
    for index, row in enumerate(rows, 1):
        assert row["sequence"] == index
        assert row["previous_sha256"] == previous
        encoded = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        previous = hashlib.sha256(encoded.encode()).hexdigest()
    assert journal.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("kind", ["tool", "tool-result", "subagent", "subagent-ends"])
def test_unstructured_tool_events_fail_closed(tmp_path: Path, kind: str) -> None:
    trajectory = Trajectory(tmp_path / "trajectory.jsonl")
    with pytest.raises(TrajectoryError, match="missing structured"):
        emit(trajectory, Evidence(kind, "a shortened description"))
    with pytest.raises(TrajectoryError):
        emit(trajectory, Evidence("result", "ignore missing evidence"))
    with pytest.raises(TrajectoryError):
        trajectory.finish(aborted=True)


def test_native_subagents_are_recorded_and_refused_for_missing_coverage(
    tmp_path: Path,
) -> None:
    journal = tmp_path / "trajectory.jsonl"
    trajectory = Trajectory(journal)
    event = tool("started", name="collabAgentToolCall")
    event.kind = "subagent"
    with pytest.raises(TrajectoryError, match="incomplete tool coverage"):
        emit(trajectory, event)
    with pytest.raises(TrajectoryError):
        trajectory.finish(aborted=True)
    assert "collabAgentToolCall" in journal.read_text()


def test_success_cannot_hide_missing_tool_results(tmp_path: Path) -> None:
    trajectory = Trajectory(tmp_path / "trajectory.jsonl")
    emit(trajectory, tool("started"))
    with pytest.raises(TrajectoryError, match="completions are missing"):
        trajectory.finish()


def test_completion_only_items_are_retained(tmp_path: Path) -> None:
    trajectory = Trajectory(tmp_path / "trajectory.jsonl")
    emit(trajectory, tool("completed", name="somethingNew"))
    trajectory.finish()


def test_unsupported_backend_is_refused_before_any_model_turn(tmp_path: Path) -> None:
    trajectory = Trajectory(tmp_path / "trajectory.jsonl")
    with pytest.raises(TrajectoryError, match="unsupported backend"):
        trajectory.opened(cast("Any", SimpleNamespace(backend="unknown")))
    with pytest.raises(TrajectoryError):
        trajectory.finish(aborted=True)


def test_journals_cannot_overwrite_existing_evidence(tmp_path: Path) -> None:
    journal = tmp_path / "trajectory.jsonl"
    journal.write_text("preserved")
    with pytest.raises(FileExistsError):
        Trajectory(journal)
    assert journal.read_text() == "preserved"


def test_failed_durability_is_not_reported_as_a_complete_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = Trajectory(tmp_path / "trajectory.jsonl")

    def failed(_: int) -> None:
        raise OSError("disk unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", failed)
        with pytest.raises(TrajectoryError, match="trajectory evidence failed"):
            emit(trajectory, tool("completed"))
    with pytest.raises(TrajectoryError):
        trajectory.finish(aborted=True)
