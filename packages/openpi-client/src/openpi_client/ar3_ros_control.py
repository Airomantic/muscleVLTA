"""Explicit live-action transport; hardware safeguards run in an independent ROS worker."""

import numpy as np

from openpi_client.ar3_ros_readonly import ReadOnlyAR3Robot
from openpi_client.ar3_ros_readonly import ROSWorkerError
from openpi_client.ar3_ros_readonly import frame_flag


class LiveAR3Robot(ReadOnlyAR3Robot):
    """Send one bounded action per fresh observation and require a matching acknowledgement."""

    read_only = False
    handles_action_limits = True

    def __init__(self, **kwargs):
        self._control_ready = False
        self._stop_confirmed = False
        self._observation_id: int | None = None
        super().__init__(**kwargs)

    def _read_frame(self, *, timeout: float | None = None) -> dict[str, np.ndarray]:
        try:
            result = super()._read_frame(timeout=timeout)
        except ROSWorkerError as exc:
            self._stop_confirmed |= exc.stopped
            raise
        self._stop_confirmed |= frame_flag(result, "__stopped__")
        return result

    def _validate_handshake(self, handshake: dict[str, np.ndarray]) -> None:
        super()._validate_handshake(handshake)
        if not frame_flag(handshake, "__control__"):
            raise RuntimeError("Live execution requires an explicitly armed control worker.")
        self._control_ready = True

    def observe(self) -> dict[str, np.ndarray]:
        self._observation_id = None
        result = super().observe()
        identifier = result.pop("__observation_id__", None)
        if identifier is None or identifier.shape != () or not np.issubdtype(identifier.dtype, np.integer):
            raise RuntimeError("Control worker did not identify its fresh observation.")
        self._observation_id = int(identifier)
        return result

    def execute(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 9D live action.")
        if self._observation_id is None:
            raise RuntimeError("Observe before each live action; observations cannot be reused.")
        identifier = self._observation_id
        self._observation_id = None
        self._send("execute", observation_id=identifier, action=action.tolist())
        result = self._read_frame()
        acknowledged = result.get("__observation_id__")
        applied = result.get("__executed_action__")
        if (
            acknowledged is None
            or acknowledged.shape != ()
            or not np.issubdtype(acknowledged.dtype, np.integer)
            or int(acknowledged) != identifier
        ):
            raise RuntimeError("Control worker acknowledged a different observation.")
        if applied is None or applied.shape != (9,) or not np.isfinite(applied).all():
            raise RuntimeError("Control worker did not acknowledge the applied 9D target.")
        return np.asarray(applied, dtype=np.float32)

    def _request_close(self) -> None:
        if self._stop_confirmed:
            return
        super()._request_close()
        if self._control_ready:
            result = self._read_frame(timeout=3)
            if not frame_flag(result, "__stopped__"):
                raise RuntimeError("Hardware stop was not acknowledged; use the physical emergency stop if needed.")

    def close(self) -> None:
        requires_stop = not self._closed and self._control_ready
        super().close()
        if requires_stop and not self._stop_confirmed:
            raise RuntimeError("Control worker exited without a stop acknowledgement; inspect the robot.")
