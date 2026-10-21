import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import REACTIVE_REQUEST_KEY
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

# Entries of the replan response that are kept on the server and passed back to `reactive_get_k_actions`.
_REACTIVE_CACHE_KEYS = ("noise", "reac_noise", "prefix_tokens", "prefix_mask", "prefix_ar_mask", "dt", "kv_cache")


class ReactiveSession:
    """Runs a force-reactive (LIFT) policy chunk by chunk for one client connection.

    `replan` encodes the chunk-start observation (images, prompt, state) once with the VLM and keeps its KV cache and
    sampling noise on the server; `step` re-runs only the action experts with the force / tactile readings recorded
    since the chunk start, so the reactive expert reacts to contact within the chunk (late reactive force injection).
    Reactive action i only attends to force / tactile readings up to step i - cross_attention_latency, so with
    k <= latency + 1 every returned action uses all readings it can see during training.
    """

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        if not (hasattr(policy, "reactive_replan") and hasattr(policy, "reactive_get_k_actions")):
            raise ValueError("The served policy does not support reactive inference.")
        self._policy = policy
        self._obs: dict | None = None
        self._cache: dict | None = None
        self._next_step = 0

    def handle(self, request: dict, obs: dict) -> dict:
        op = request.get("op")
        k = int(request.get("k", 1))
        if op == "replan":
            self._obs = dict(obs)
            self._cache = self._policy.reactive_replan(self._obs)
            return self._get_k_actions({}, k, current_step=0)
        if op == "step":
            if self._cache is None:
                raise ValueError("Reactive 'step' requires a preceding 'replan'.")
            return self._get_k_actions(obs, k, current_step=int(request.get("current_step", self._next_step)))
        raise ValueError(f"Unknown reactive op {op!r}, expected 'replan' or 'step'.")

    def _get_k_actions(self, readings: dict, k: int, *, current_step: int) -> dict:
        assert self._cache is not None
        assert self._obs is not None
        action_horizon = int(self._cache["action_horizon"])
        if not 0 <= current_step < action_horizon:
            raise ValueError(f"current_step must be in [0, {action_horizon}), got {current_step}; replan a new chunk.")
        start_time = time.monotonic()
        outputs = self._policy.reactive_get_k_actions(
            obs={**self._obs, **readings},
            k=k,
            current_step=current_step,
            sequence_id=self._cache["sequence_id"],
            **{key: self._cache[key] for key in _REACTIVE_CACHE_KEYS},
        )
        self._next_step = int(outputs["next_step"])
        # The policy pads the k actions to the action horizon; return only the actions of the requested steps.
        outputs["actions"] = outputs["actions"][: self._next_step - current_step]
        outputs["start_step"] = current_step
        outputs["action_horizon"] = action_horizon
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - start_time) * 1000}
        return outputs


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Implements `infer` and, for force-reactive policies, chunk-wise reactive inference (see ReactiveSession): requests
    carrying the REACTIVE_REQUEST_KEY entry {"op": "replan" | "step", "k": ...} are routed to a per-connection session.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        reactive_session: ReactiveSession | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                reactive_request = obs.pop(REACTIVE_REQUEST_KEY, None)

                infer_time = time.monotonic()
                if reactive_request is None:
                    action = self._policy.infer(obs)
                else:
                    if reactive_session is None:
                        reactive_session = ReactiveSession(self._policy)
                    action = reactive_session.handle(reactive_request, obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
