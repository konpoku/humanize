"""Public access to emitted trajectory recording, implemented in the tracing layer."""

from hmz.runtime.tracing.trajectory import Trajectory, TrajectoryError

__all__ = ["Trajectory", "TrajectoryError"]
