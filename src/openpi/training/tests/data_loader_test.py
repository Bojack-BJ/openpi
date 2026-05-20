import dataclasses

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_parse_subtask_segments_from_boundaries_payload():
    payload = {
        "boundaries_frame_indices": [168, 462],
        "subtask": ["pick up the sponge", "place the sponge in the grid", "return home"],
        "subtask_instruction": [
            {"[0,167]": "pick up the sponge"},
            {"[168,461]": "place the sponge in the grid"},
            {"[462,900]": "return home"},
        ],
    }

    segments = _data_loader._parse_subtask_segments(payload)

    assert segments[0] == (
        (0, 168, "pick up the sponge"),
        (168, 462, "place the sponge in the grid"),
        (462, 901, "return home"),
    )


def test_parse_subtask_segments_from_episode_mapping():
    payload = {
        "episodes": {
            "3": {
                "segments": [
                    {"start_frame": 0, "end_frame": 5, "subtask": "pick"},
                    {"start_frame": 5, "end_frame": 9, "subtask": "place"},
                ]
            }
        }
    }

    segments = _data_loader._parse_subtask_segments(payload)

    assert segments[3] == ((0, 5, "pick"), (5, 9, "place"))


def test_parse_subtask_segments_from_episode_list():
    payload = {
        "episodes": [
            {
                "episode_index": 2,
                "segments": [
                    {"start_frame": 1, "end_frame": 4, "subtask": "align"},
                ],
            }
        ]
    }

    segments = _data_loader._parse_subtask_segments(payload)

    assert segments[2] == ((1, 4, "align"),)


def test_load_subtask_segments_missing_file_returns_empty(tmp_path):
    segments = _data_loader._load_subtask_segments("missing_subtask_segments.json", tmp_path)

    assert segments == {}
