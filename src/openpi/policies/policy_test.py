import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy

from openpi.models import model
from openpi.policies import ar3_force_visuotactile_policy as ar3
from openpi.policies import single_iphone_flexiv_policy


class CountingPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.calls = 0

    def infer(self, obs):
        self.calls += 1
        return {"actions": np.arange(28).reshape(4, 7)}

    def reset(self):
        self.calls = 0


def test_single_arm_policy_input_output():
    example = single_iphone_flexiv_policy.make_single_iphone_flexiv_example()
    inputs = single_iphone_flexiv_policy.SingleiPhoneFlexivInputs(model_type=model.ModelType.PI05)(example)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert inputs["image_mask"]["left_wrist_0_rgb"]
    assert not inputs["image_mask"]["base_0_rgb"]
    np.testing.assert_array_equal(inputs["wrench"], example["wrench"])
    outputs = single_iphone_flexiv_policy.SingleiPhoneFlexivOutputs()({"actions": np.zeros((10, 32))})
    assert outputs["actions"].shape == (10, 7)


def test_action_chunk_broker_refresh_and_reset():
    policy = CountingPolicy()
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)
    first = broker.infer({})
    second = broker.infer({})
    assert policy.calls == 1
    np.testing.assert_array_equal(first["actions"], np.arange(7))
    np.testing.assert_array_equal(second["actions"], np.arange(7, 14))
    broker.infer({})
    assert policy.calls == 2
    broker.reset()
    assert policy.calls == 0
    np.testing.assert_array_equal(broker.infer({})["actions"], np.arange(7))


def test_ar3_force_level_appended_after_normalization():
    inputs = ar3.AR3ForceVisuotactileInputs(model_type=model.ModelType.PI05, action_horizon=3)
    actions = np.zeros((3, ar3.ACTION_DIM), dtype=np.float32)
    actions[:, ar3.FORCE_LEVEL_ACTION_INDEX] = [20.0, 60.0, 100.0]
    data = {
        ar3.FRONT_IMAGE_KEY: np.zeros((8, 8, 3), np.uint8),
        ar3.WRIST_IMAGE_KEY: np.zeros((8, 8, 3), np.uint8),
        ar3.STATE_KEY: np.zeros(27, np.float32),
        ar3.ARM_EFFORT_KEY: np.zeros((3, 7), np.float32),
        ar3.FINGER_WRENCH_KEYS[0]: np.zeros((3, 6), np.float32),
        ar3.FINGER_WRENCH_KEYS[1]: np.zeros((3, 6), np.float32),
        "actions": actions,
    }
    out = ar3.AR3AppendForceLevel()(inputs(data))
    assert out["wrench"].shape == (3, ar3.MODEL_FORCE_DIM)
    np.testing.assert_allclose(out["wrench"][:, -1], [-1.0, 0.0, 1.0])
    # Inference: a single commanded level is repeated over the horizon.
    del data["actions"]
    out = ar3.AR3AppendForceLevel()(inputs({**data, ar3.FORCE_LEVEL_KEY: 80.0}))
    np.testing.assert_allclose(out["wrench"][:, -1], 0.5)
