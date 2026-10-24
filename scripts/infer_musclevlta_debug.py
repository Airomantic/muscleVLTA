"""Local-model debug: read-only by default; --source ros --execute enables physical actions."""

import copy
import json
import logging
import os
from pathlib import Path
import time

import ar3_force_visuotactile_client_example as client_example

LOWMEM_CONFIG = "pi05_ar3_force_visuotactile_reactive_lowmem"
FINE_TUNING_CONFIG = "pi05_ar3_force_visuotactile_reactive"


class LocalPolicyClient:
    """Same reactive protocol as serving, but in the debugger's process."""

    def __init__(self, policy, metadata):
        from openpi.serving.websocket_policy_server import ReactiveSession

        self._policy = policy
        self._metadata = metadata
        self._session = ReactiveSession(policy)

    def get_server_metadata(self):
        return self._metadata

    def infer(self, observation):
        return self._policy.infer(observation)

    def _reactive(self, request, observation):
        start = time.monotonic()
        result = self._session.handle(request, observation)
        result["server_timing"] = {"infer_ms": (time.monotonic() - start) * 1000}
        return result

    def reactive_replan(self, observation, *, k):
        return self._reactive({"op": "replan", "k": k}, observation)

    def reactive_step(self, readings, *, k):
        return self._reactive({"op": "step", "k": k}, readings)


def check_readonly_observations(args):
    logging.info("Checking real ROS observations before loading the model; no hardware commands will be sent.")
    readonly_args = copy.copy(args)
    readonly_args.source = "ros-readonly"
    readonly_args.execute = False
    readonly_args.initialize_gripper = False
    robot = client_example.make_robot(readonly_args, {})
    try:
        observation = robot.observe()
    finally:
        robot.close()
    logging.info(
        "Observation check passed: front=%s, wrist=%s, gripper position=%g SDK units.",
        observation[client_example.FRONT_IMAGE_KEY].shape,
        observation[client_example.WRIST_IMAGE_KEY].shape,
        observation[client_example.STATE_KEY][-1],
    )


def lora_mismatch(checkpoint: Path, config_name: str, model_config) -> str | None:
    """Error message if the checkpoint and the model config disagree on LoRA adapters, checked before loading."""
    try:
        tree = json.loads((checkpoint / "params" / "_METADATA").read_text())["tree_metadata"]
    except (OSError, ValueError, KeyError, TypeError):
        tree = None
    if not isinstance(tree, dict) or not any("PaliGemma" in key for key in tree):
        return None  # Unknown layout: the model load still verifies the full parameter structure.
    has_lora = any("lora" in key for key in tree)
    uses_lora = any(
        "lora" in str(getattr(model_config, name, ""))
        for name in ("paligemma_variant", "action_expert_variant", "reactive_action_expert_variant")
    )
    if has_lora == uses_lora:
        return None
    kind, expected = ("LoRA (lowmem) training", LOWMEM_CONFIG) if has_lora else ("full fine-tuning", FINE_TUNING_CONFIG)
    return (
        f"{checkpoint} comes from {kind}, but --config-name {config_name} "
        f"{'lacks' if has_lora else 'adds'} LoRA adapters; use --config-name {expected}."
    )


def main(argv=None):
    parser = client_example.build_parser()
    parser.description = __doc__
    parser.set_defaults(source="ros-readonly", mode="reactive", rerun="spawn", fps=15.0)
    parser.add_argument("--checkpoint-dir", default=str(Path(__file__).resolve().parents[1] / "infer_checkpoint/29999"))
    parser.add_argument("--config-name", default="pi05_ar3_force_visuotactile_reactive_lowmem")
    parser.add_argument("--gpu", default="0", help="visible GPU index or cpu")
    parser.add_argument(
        "--check-observations",
        action="store_true",
        help="Check real read-only observations and exit without loading a model or opening Rerun",
    )
    args = parser.parse_args(argv)
    if args.source not in ("ros-readonly", "ros"):
        parser.error("This debug entrypoint requires --source ros-readonly or --source ros --execute.")
    if args.check_observations and args.source != "ros-readonly":
        parser.error("--check-observations only allows --source ros-readonly.")
    try:
        client_example.validate_execution_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.execute:
        for path in (
            Path(__file__).with_name("ar3_ros_control_worker.py"),
            Path(args.ros_python),
            Path(args.ros_setup),
        ):
            if not path.is_file():
                parser.error(f"Missing live control prerequisite: {path}")

    logging.warning(
        "LIVE DEBUG: the arm and gripper WILL MOVE; keep the emergency stop accessible."
        if args.execute
        else "READ-ONLY DEBUG: no arm/gripper actions or initialization are sent."
    )
    if args.initialize_gripper:
        logging.info(
            "Full observation preflight is deferred to the LIVE worker because motor initialization was explicitly "
            "requested; valid observations will still be required before any policy action."
        )
    else:
        check_readonly_observations(args)
    if args.check_observations:
        return None

    # Configure JAX before importing the model stack.
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == "cpu" else args.gpu
    os.environ["JAX_PLATFORMS"] = "cpu" if args.gpu == "cpu" else "cuda"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    assets = list((checkpoint / "assets").glob("*/norm_stats.json"))
    if "OPENPI_AR3_ASSET_ID" not in os.environ:
        if len(assets) != 1:
            parser.error("Expected one checkpoint normalization asset; set OPENPI_AR3_ASSET_ID.")
        os.environ["OPENPI_AR3_ASSET_ID"] = assets[0].parent.name

    from serve_policy import server_metadata

    from openpi.policies import policy_config
    from openpi.training import config

    logging.info("Do not run a second policy server on this GPU at the same time.")
    train_config = config.get_config(args.config_name)
    if error := lora_mismatch(checkpoint, args.config_name, train_config.model):
        parser.error(error)
    policy = policy_config.create_trained_policy(
        train_config, checkpoint, default_prompt=args.prompt or client_example.PROMPT
    )
    return client_example.run(args, policy_client=LocalPolicyClient(policy, server_metadata(train_config)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
