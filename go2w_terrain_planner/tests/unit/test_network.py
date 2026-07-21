import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.models import ActorExportWrapper, Go2wActorCritic
from go2w_terrain_planner.models.map_encoder import MapEncoder


def test_map_encoder_preserves_coarse_spatial_layout() -> None:
    torch.manual_seed(0)
    encoder = MapEncoder(input_channels=4, feature_dim=16).eval()
    left = torch.zeros((1, 4, 64, 64))
    right = torch.zeros_like(left)
    left[:, 1, 24:32, 8:16] = 1.0
    right[:, 1, 24:32, 40:48] = 1.0

    left_feature = encoder(left)
    right_feature = encoder(right)

    assert left_feature.shape == (1, 16)
    assert not torch.allclose(left_feature, right_feature, atol=1.0e-6)


def test_actor_critic_forward_and_action_bounds() -> None:
    history, channels, size, auxiliary = 3, 4, 16, 11
    policy_dim = history * channels * size * size + auxiliary
    obs = {"policy": torch.zeros((2, policy_dim)), "critic": torch.zeros((2, 7))}
    model = Go2wActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_history_length=history,
        map_channels=channels,
        map_size=size,
        critic_hidden_dims=[16],
    )
    action = model.act_inference(obs)
    sampled_action = model.act(obs)
    log_probability = model.get_actions_log_prob(sampled_action)
    value = model.evaluate(obs)
    assert action.shape == (2, 2)
    assert value.shape == (2, 1)
    assert torch.max(torch.abs(action)).item() <= 1.0
    assert torch.max(torch.abs(sampled_action)).item() < 1.0
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(model.entropy).all()
    assert torch.equal(model.action_mean, model.distribution.mean)
    wrapper = ActorExportWrapper(model, (-0.2, -1.0), (0.8, 1.0))
    command = wrapper(obs["policy"])
    assert command.shape == (2, 2)
    assert torch.all(command[:, 0] >= -0.2) and torch.all(command[:, 0] <= 0.8)
    assert torch.all(command[:, 1] >= -1.0) and torch.all(command[:, 1] <= 1.0)


def test_onnx_wrapper_supports_dynamic_batch(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    history, channels, size, auxiliary = 3, 4, 16, 11
    policy_dim = history * channels * size * size + auxiliary
    observations = {
        "policy": torch.zeros((1, policy_dim)),
        "critic": torch.zeros((1, 7)),
    }
    model = Go2wActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_history_length=history,
        map_channels=channels,
        map_size=size,
        critic_hidden_dims=[16],
    )
    wrapper = ActorExportWrapper(model, (-0.2, -1.0), (0.8, 1.0)).eval()
    output = tmp_path / "actor.onnx"
    torch.onnx.export(
        wrapper,
        torch.zeros((1, policy_dim)),
        output,
        input_names=["policy_observation"],
        output_names=["velocity_command"],
        dynamic_axes={"policy_observation": {0: "batch"}, "velocity_command": {0: "batch"}},
        opset_version=17,
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    action = session.run(
        None, {"policy_observation": np.zeros((2, policy_dim), dtype=np.float32)}
    )[0]
    assert action.shape == (2, 2)
    assert np.isfinite(action).all()
