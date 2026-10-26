#!/usr/bin/env python3
"""Explicitly prepare the server's CAN connection for inference, then exit.

--connect authorizes Connect only when the server reports disconnected. Optional
--initialize additionally allows Enable and full calibration, which moves the
gripper. No target, open, close, or Disconnect RPC is sent. Closing this client's
gRPC channel preserves the server's CAN connection for subsequent inference.
"""

import argparse
from collections.abc import Callable, Sequence
import math
import sys
import time
from typing import Any

if __package__:
    from scripts import ar3_ros_observation_worker as observations
else:
    import ar3_ros_observation_worker as observations


MAX_AGE = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", action="store_true", required=True, help="Allow server CAN connection")
    parser.add_argument("--gripper-address", default="192.168.127.10:50008")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--initialize", action="store_true", help="Allow calibration if needed; moves the gripper")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    if not 1 <= args.motor_id <= 255:
        parser.error("--motor-id must be 1..255")
    if not args.gripper_address.strip() or not args.channel.strip():
        parser.error("--gripper-address and --channel must be nonempty")
    return args


def describe_state(state: Any) -> str:
    fields = ("connected", "valid", "stale", "init_state", "error", "position", "update_age_ms", "message")
    return ", ".join(f"{name}={getattr(state, name, '<missing>')}" for name in fields)


def prepare(
    args: argparse.Namespace,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not args.connect:
        raise ValueError("CAN preparation requires explicit --connect")
    from DM_V3Gripper import gripper_pb2
    import grpc

    deadline = clock() + args.timeout
    channel = grpc.insecure_channel(args.gripper_address, options=(("grpc.enable_http_proxy", 0),))
    attempted_initialization = False
    try:
        # Reject incompatible feedback schemas before binding any write RPC.
        reader = observations.bind_gripper_reader(channel, gripper_pb2.StateResponse)
        operations = {}

        def call(name: str, *, until: float, request: bytes = b"", limit: float = 5.0) -> None:
            remaining = until - clock()
            if remaining <= 0:
                raise TimeoutError(f"Deadline expired before gripper {name}")
            if name not in operations:
                operations[name] = channel.unary_unary(
                    f"/gripper.GripperService/{name}", response_deserializer=gripper_pb2.Response.FromString
                )
            try:
                response = operations[name](request, timeout=min(limit, remaining))
            except grpc.RpcError as exc:
                raise RuntimeError(f"Gripper {name} RPC failed; delivery is uncertain: {exc}") from exc
            if not response.success:
                raise RuntimeError(f"Gripper {name} rejected: {response.message}")

        inspected_connection = False
        initialization_deadline = None
        detail = "No GetAllState response received"
        while True:
            until = min(deadline, initialization_deadline) if initialization_deadline is not None else deadline
            remaining = until - clock()
            if remaining <= 0:
                phase = "initialization" if initialization_deadline is not None else "CAN feedback"
                raise TimeoutError(f"Gripper {phase} timed out: {detail}")
            started = clock()
            try:
                state = reader(b"", timeout=min(1.0, remaining))
            except grpc.RpcError as exc:
                detail = f"GetAllState failed: {exc}"
            else:
                elapsed = clock() - started
                detail = describe_state(state)
                if clock() >= until:
                    continue
                if not inspected_connection:
                    inspected_connection = True
                    if not state.connected:
                        config = gripper_pb2.ConnectConfig(
                            channel=args.channel,
                            motor_id=args.motor_id,
                            read_interval_s=0.05,
                            read_timeout_s=0.1,
                            retry_count=2,
                            stale_timeout_s=MAX_AGE,
                            command_min_interval_s=0.02,
                        )
                        print(f"Connecting gripper CAN {args.channel}, motor {args.motor_id}...", flush=True)
                        call("Connect", until=deadline, request=config.SerializeToString())
                        continue
                age = float(state.update_age_ms) / 1000.0 + elapsed
                fresh = math.isfinite(age) and state.update_age_ms >= 0 and elapsed >= 0 and age <= MAX_AGE
                if state.connected and state.valid and not state.stale and fresh:
                    if state.error != 0:
                        raise RuntimeError(f"Gripper reports a motor error: {detail}")
                    if state.init_state == 1:
                        observations.gripper_position(state, max_age=MAX_AGE, elapsed=elapsed)
                        print(
                            f"Gripper CAN ready at {args.gripper_address}: position={state.position}. "
                            "Closing only the local gRPC channel; server CAN connection retained for inference. "
                            "No target command was sent.",
                            flush=True,
                        )
                        return
                    if state.init_state not in (0, 2):
                        raise RuntimeError(f"Unknown gripper initialization state: {detail}")
                    if state.init_state == 0 and not args.initialize:
                        raise RuntimeError(
                            "Gripper CAN is connected but not initialized. "
                            "Run ./infer_oneclick.sh --ready-only --initialize-gripper "
                            "to explicitly allow calibration motion."
                        )
                    if initialization_deadline is None:
                        initialization_deadline = min(deadline, clock() + 15.0)
                        if state.init_state == 0:
                            print("Initializing gripper; calibration moves the fingers...", flush=True)
                            # Even a rejected/timed-out enable may have reached the motor.
                            attempted_initialization = True
                            call("Enable", until=initialization_deadline)
                            call("Initialization", until=initialization_deadline)
                        else:
                            print("Waiting for gripper initialization already in progress...", flush=True)
            sleep(max(0.0, min(0.05, until - clock())))
    except BaseException:
        if attempted_initialization:
            try:
                call("Stop", until=clock() + 2.0, limit=2.0)
            except BaseException as exc:
                print(f"Gripper cleanup Stop failed; physical stop is unverified: {exc}", file=sys.stderr, flush=True)
            else:
                print(
                    "Gripper cleanup Stop RPC acknowledged; physical stop is unverified.", file=sys.stderr, flush=True
                )
        raise
    finally:
        channel.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare(args)
    except KeyboardInterrupt:
        print("Gripper preparation interrupted.", file=sys.stderr)
        return 130
    except (ImportError, RuntimeError, TimeoutError, ValueError, AttributeError) as exc:
        print(f"Gripper preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
