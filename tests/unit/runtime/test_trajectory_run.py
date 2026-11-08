"""A mandatory recording failure cancels the root before workflow continuation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from hmz.runtime import Run
from hmz.runtime.trajectory import TrajectoryError
from tests.unit.runtime.test_trajectory import Evidence, tool

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class ObservedAgent:
    id: str = "worker"
    backend: str = "codex"
    observers: list[Any] = field(default_factory=list[Any])

    def watch(self, callback: Any, *, critical: bool = False) -> None:
        assert critical
        self.observers.append(callback)

    def watch_critical(self, callback: Any) -> None:
        self.watch(callback, critical=True)

    def emit(self, session: object, event: Evidence) -> None:
        for observer in self.observers:
            observer(self, session, event)


@dataclass
class ScriptedRunner:
    agent: ObservedAgent = field(default_factory=ObservedAgent)
    steps: list[str] = field(default_factory=list[str])
    incomplete: bool = False

    @property
    def agents(self) -> dict[str, ObservedAgent]:
        return {"worker": self.agent}

    async def arun(self, task: str, **callbacks: Any) -> str:
        session = SimpleNamespace()
        callbacks["opened"]("worker", self.agent, session, None)
        await asyncio.to_thread(self.agent.emit, session, Evidence("begins", task))
        event = (
            Evidence("tool", "a truncated label")
            if self.incomplete
            else tool("started")
        )
        await asyncio.to_thread(self.agent.emit, session, event)
        self.steps.append("continued after tool")
        await asyncio.to_thread(self.agent.emit, session, tool("completed"))
        return "success"


def test_missing_tool_evidence_cancels_the_entire_run_before_continuation(
    tmp_path: Path,
) -> None:
    runner = ScriptedRunner(incomplete=True)
    run = Run(cast("Any", runner), "fixed prompt")
    journal = tmp_path / "trajectory.jsonl"
    run.record_trajectory(journal)

    with pytest.raises(TrajectoryError, match="missing structured"):
        run.run()

    assert runner.steps == []
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["status"] == "failed"


def test_a_complete_run_records_both_tool_notifications(tmp_path: Path) -> None:
    runner = ScriptedRunner()
    run = Run(cast("Any", runner), "fixed prompt")
    run.record_trajectory(tmp_path / "trajectory.jsonl")
    assert run.run() == "success"
    assert runner.steps == ["continued after tool"]


def test_recording_cannot_be_replaced_after_attachment(tmp_path: Path) -> None:
    run = Run(cast("Any", ScriptedRunner()), "fixed prompt")
    run.record_trajectory(tmp_path / "one.jsonl")
    with pytest.raises(RuntimeError, match="once, before starting"):
        run.record_trajectory(tmp_path / "two.jsonl")
    assert run.run() == "success"


def test_offline_mode_requires_each_driver_to_support_enforcement() -> None:
    run = Run(cast("Any", SimpleNamespace(agents={"worker": object()})), "task")
    with pytest.raises(RuntimeError, match="cannot enforce offline"):
        run.offline()
