import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Reserved request key selecting a reactive (LIFT) operation instead of a plain `infer` call.
REACTIVE_REQUEST_KEY = "__reactive__"


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.

    Besides `infer` (one action chunk per call), force-reactive (LIFT) models can be run chunk by chunk with
    `reactive_replan` and `reactive_step`: the server encodes images, prompt and state once per chunk and then re-runs
    only the action experts with the newest force / tactile readings every k steps.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        return self._request(obs)

    def reactive_replan(self, obs: Dict, *, k: int) -> Dict:  # noqa: UP006
        """Starts a new action chunk of a force-reactive model and returns its first k actions.

        `obs` is a full observation (as for `infer`) taken at the start of the chunk. The response contains
        `actions` (k, action_dim), `start_step` (0), `next_step` and `action_horizon`.
        """
        return self._request({**obs, REACTIVE_REQUEST_KEY: {"op": "replan", "k": k}})

    def reactive_step(self, readings: Dict, *, k: int, current_step: Optional[int] = None) -> Dict:  # noqa: UP006
        """Returns the next k actions of the current chunk, re-planned with the newest force / tactile readings.

        `readings` holds the force / tactile observations recorded since the start of the chunk, one reading per
        executed step (leading dimension = number of readings). They replace the corresponding entries of the
        observation sent to `reactive_replan`; images, prompt and state stay those of the chunk start.
        `current_step` defaults to the `next_step` of the previous response.
        """
        request = {"op": "step", "k": k}
        if current_step is not None:
            request["current_step"] = current_step
        return self._request({**readings, REACTIVE_REQUEST_KEY: request})

    def _request(self, data: Dict) -> Dict:  # noqa: UP006
        self._ws.send(self._packer.pack(data))
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
