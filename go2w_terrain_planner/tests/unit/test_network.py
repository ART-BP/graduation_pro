import pytest

torch = pytest.importorskip("torch")

from go2w_terrain_planner.models import ActorExportWrapper, Go2wActorCritic
from go2w_terrain_planner.models.map_encoder import CompactTerrainMapEncoder


def test_compact_encoder_preserves_spatial_layout_and_change_layer() -> None:
    torch.manual_seed(1)
    encoder = CompactTerrainMapEncoder(
        input_channels=7,
        map_size=32,
        feature_dim=48,
        encoder_channels=(8, 16, 24, 32),
        spatial_pool_size=4,
    ).eval()
    maps = torch.zeros((1, 7, 32, 32))
    maps[:, 1, 12:18, 5:10] = 1.0
    maps[:, 2:4, 12:18, 5:10] = 1.0
    goal = torch.tensor([[0.6, 0.0, 1.0]])

    original = encoder(maps, goal)
    mirrored_maps = torch.flip(maps, dims=(-1,))
    mirrored = encoder(mirrored_maps, goal)
    changed = maps.clone()
    changed[:, 4, 12:18, 5:10] = 1.0
    changed_feature = encoder(changed, goal)

    assert original.shape == (1, 48)
    assert not torch.allclose(original, mirrored, atol=1.0e-6)
    assert not torch.allclose(original, changed_feature, atol=1.0e-6)


def test_compact_encoder_propagates_gradient_to_all_channels() -> None:
    torch.manual_seed(2)
    encoder = CompactTerrainMapEncoder(
        input_channels=7,
        map_size=16,
        feature_dim=24,
        encoder_channels=(8, 16, 24, 32),
        spatial_pool_size=2,
    )
    maps = torch.randn((2, 7, 16, 16), requires_grad=True)
    goal = torch.tensor([[0.5, 0.0, 1.0], [0.7, 1.0, 0.0]])

    encoder(maps, goal).square().mean().backward()

    assert maps.grad is not None
    per_channel_gradient = maps.grad.abs().sum(dim=(0, 2, 3))
    assert torch.all(per_channel_gradient > 0.0)


def test_actor_critic_forward_and_action_bounds() -> None:
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    obs = {"policy": torch.zeros((2, policy_dim)), "critic": torch.zeros((2, 7))}
    model = Go2wActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
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
    assert model.action_std.mean().item() == pytest.approx(0.4)
    wrapper = ActorExportWrapper(model, (-0.2, -1.0), (0.8, 1.0))
    command = wrapper(obs["policy"])
    assert command.shape == (2, 2)
    assert torch.all(command[:, 0] >= -0.2) and torch.all(command[:, 0] <= 0.8)
    assert torch.all(command[:, 1] >= -1.0) and torch.all(command[:, 1] <= 1.0)
    with torch.no_grad():
        model.actor_head[-1].weight.zero_()
        model.actor_head[-1].bias.zero_()
    assert torch.equal(wrapper(obs["policy"]), torch.zeros((2, 2)))


def test_saturated_actor_head_keeps_distribution_and_log_probability_finite() -> None:
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    obs = {"policy": torch.zeros((2, policy_dim)), "critic": torch.zeros((2, 7))}
    model = Go2wActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
        critic_hidden_dims=[16],
        maximum_pre_tanh_mean=2.5,
    )
    with torch.no_grad():
        model.actor_head[-1].weight.zero_()
        model.actor_head[-1].bias.fill_(100.0)

    deterministic_action = model.act_inference(obs)
    model.act(obs)
    boundary_actions = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    log_probability = model.get_actions_log_prob(boundary_actions)

    assert torch.max(torch.abs(model.action_mean)).item() <= 2.5
    assert torch.max(torch.abs(deterministic_action)).item() < 0.99
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(model.entropy).all()


def test_action_std_has_gradient_near_configured_upper_bound() -> None:
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    obs = {"policy": torch.zeros((2, policy_dim)), "critic": torch.zeros((2, 7))}
    model = Go2wActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
        critic_hidden_dims=[16],
    )
    with torch.no_grad():
        model.std.fill_(5.0)

    bounded_std = model._bounded_action_std()
    bounded_std.sum().backward()

    assert torch.all(bounded_std < model.maximum_action_std)
    assert torch.all(bounded_std > model.minimum_action_std)
    assert torch.isfinite(model.std.grad).all()
    assert torch.all(model.std.grad.abs() > 0.0)


def test_squashed_entropy_propagates_gradient_to_actor_mean() -> None:
    torch.manual_seed(3)
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    obs = {"policy": torch.zeros((8, policy_dim)), "critic": torch.zeros((8, 7))}
    model = Go2wActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
        critic_hidden_dims=[16],
    )

    model.act(obs)
    (-model.entropy.mean()).backward()

    actor_bias_gradient = model.actor_head[-1].bias.grad
    assert actor_bias_gradient is not None
    assert torch.isfinite(actor_bias_gradient).all()
    assert torch.any(actor_bias_gradient.abs() > 0.0)


def test_motion_gru_preserves_sequence_order_and_receives_gradient() -> None:
    torch.manual_seed(9)
    history, channels, size = 2, 7, 16
    auxiliary = 3 + 2 + 2 * history + 3 * history
    policy_dim = channels * size * size + auxiliary
    first = torch.zeros((1, policy_dim))
    second = first.clone()
    map_end = channels * size * size
    motion_start = map_end + 3 + 2 + 2 * history
    first[:, motion_start:] = torch.tensor([[0.8, 0.0, 0.2, -0.3, 0.1, -0.4]])
    second[:, motion_start:] = first[:, motion_start:].reshape(1, history, 3).flip(1).flatten(1)
    model = Go2wActorCritic(
        {"policy": first, "critic": torch.zeros((1, 7))},
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
        critic_hidden_dims=[16],
    )

    first_mean = model._actor_raw_mean(first)
    second_mean = model._actor_raw_mean(second)
    first_mean.sum().backward()

    assert not torch.allclose(first_mean, second_mean, atol=1.0e-6)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.motion_gru.parameters()
    )


def test_removed_full_map_stack_architecture_is_rejected() -> None:
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    obs = {"policy": torch.zeros((1, policy_dim)), "critic": torch.zeros((1, 7))}
    with pytest.raises(ValueError, match="compact_map_motion_gru_v7"):
        Go2wActorCritic(
            obs,
            {"policy": ["policy"], "critic": ["critic"]},
            2,
            map_channels=channels,
            map_size=size,
            command_history_length=history,
            motion_history_length=history,
            architecture="spatial_temporal_fpn_v5",
            critic_hidden_dims=[16],
        )


def test_onnx_wrapper_supports_dynamic_batch(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    history, channels, size, auxiliary = 2, 7, 16, 15
    policy_dim = channels * size * size + auxiliary
    observations = {
        "policy": torch.zeros((1, policy_dim)),
        "critic": torch.zeros((1, 7)),
    }
    model = Go2wActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        2,
        map_channels=channels,
        map_size=size,
        command_history_length=history,
        motion_history_length=history,
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
