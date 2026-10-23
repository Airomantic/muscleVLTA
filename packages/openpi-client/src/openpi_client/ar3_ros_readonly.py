"""Private-pipe transport to a ROS Jazzy worker; this adapter cannot command hardware."""

import io
import json
import logging
import math
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)
MAX_FRAME_BYTES = 64 * 1024 * 1024


def frame_flag(frame: dict[str, np.ndarray], key: str) -> bool:
    value = frame.get(key)
    return value is not None and value.shape == () and value.dtype == np.bool_ and bool(value.item())


class ROSWorkerError(RuntimeError):
    def __init__(self, message: str, *, stopped: bool = False):
        super().__init__(f"ROS observation worker: {message}")
        self.stopped = stopped


class ReadOnlyAR3Robot:
    """Read real observations in a separate Python/ROS environment; discard proposed actions."""

    read_only = True

    def __init__(
        self,
        *,
        worker_script: str | Path,
        ros_python: str | Path,
        ros_setup: str | Path,
        domain_id: int = 111,
        timeout: float = 15.0,
        startup_timeout: float | None = None,
        worker_args: Sequence[str] = (),
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("ROS worker timeout must be finite and positive.")
        if startup_timeout is not None and (not math.isfinite(startup_timeout) or startup_timeout <= 0):
            raise ValueError("ROS worker startup timeout must be finite and positive.")
        if not 0 <= domain_id <= 232:
            raise ValueError("ROS domain ID must be 0..232.")
        for path in (worker_script, ros_python, ros_setup):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing ROS worker prerequisite: {path}")
        self._timeout = timeout
        self._closed = False
        self.last_prediction: np.ndarray | None = None
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"HOME", "USER", "LOGNAME", "LANG", "TERM", "DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR"}
        }
        environment.update(
            PATH="/usr/bin:/bin:/usr/local/bin",
            ROS_DOMAIN_ID=str(domain_id),
            PYTHONUNBUFFERED="1",
            PYTHONNOUSERSITE="1",
            RCUTILS_LOGGING_USE_STDOUT="0",
        )
        command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -e; source "$1"; shift; exec "$@"',
            "_",
            str(ros_setup),
            str(ros_python),
            "-u",
            str(worker_script),
            *worker_args,
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
            env=environment,
            start_new_session=not self.read_only,
        )
        assert self._process.stdout is not None
        self._selector = selectors.DefaultSelector()
        self._selector.register(self._process.stdout, selectors.EVENT_READ)
        ready = False
        try:
            self._validate_handshake(self._read_frame(timeout=startup_timeout))
            ready = True
        finally:
            if not ready:
                self.close()
        if self.read_only:
            logger.warning("READ-ONLY: real observations, predicted actions only; no robot or gripper commands.")
        else:
            logger.warning("LIVE EXECUTION: model actions will be sent to the arm and gripper.")

    def _validate_handshake(self, handshake: dict[str, np.ndarray]) -> None:
        if not frame_flag(handshake, "__ready__"):
            raise RuntimeError("ROS worker did not return its readiness handshake.")

    def _read_exact(self, size: int, deadline: float) -> bytes:
        assert self._process.stdout is not None
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._selector.select(remaining):
                raise TimeoutError("ROS observation worker timed out; inspect its errors in this terminal.")
            chunk = os.read(self._process.stdout.fileno(), min(size - len(result), 1024 * 1024))
            if not chunk:
                raise EOFError(f"ROS observation worker closed its pipe (exit={self._process.poll()}); see stderr.")
            result.extend(chunk)
        return bytes(result)

    def _read_frame(self, *, timeout: float | None = None) -> dict[str, np.ndarray]:
        deadline = time.monotonic() + (self._timeout if timeout is None else timeout)
        size = struct.unpack("!Q", self._read_exact(8, deadline))[0]
        if not 0 < size <= MAX_FRAME_BYTES:
            raise ValueError(f"Invalid ROS worker frame size: {size}")
        with np.load(io.BytesIO(self._read_exact(size, deadline)), allow_pickle=False) as archive:
            result = {key: archive[key] for key in archive.files}
        if "__error__" in result:
            raise ROSWorkerError(str(result["__error__"].item()), stopped=frame_flag(result, "__stopped__"))
        return result

    def _send(self, operation: str, **fields) -> None:
        if self._closed:
            raise RuntimeError("The read-only observation adapter is closed.")
        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps({"op": operation, **fields}, allow_nan=False) + "\n").encode())
        self._process.stdin.flush()

    def observe(self) -> dict[str, np.ndarray]:
        self._send("observe")
        return self._read_frame()

    def execute(self, action: np.ndarray) -> None:
        """Keep a preview locally. Deliberately no request is sent to the ROS worker."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite 9D action preview.")
        self.last_prediction = action.copy()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.poll() is None:
                try:
                    self._request_close()
                except BrokenPipeError as exc:
                    if not self.read_only:
                        raise RuntimeError("Control worker disappeared before acknowledging a hardware stop.") from exc
                    logger.warning("ROS worker exited before accepting the close request.")
                finally:
                    try:
                        self._process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("Stopping unresponsive owned ROS worker PID %s.", self._process.pid)
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            logger.error(
                                "Killing unresponsive owned ROS worker PID %s; check hardware.", self._process.pid
                            )
                            self._process.kill()
                            self._process.wait()
        finally:
            self._closed = True
            self._selector.close()
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()

    def _request_close(self) -> None:
        self._send("close")
