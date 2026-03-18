from collections.abc import Callable
from pathlib import Path

import datasets
from datasets import load_dataset
import numpy as np
import packaging.version
import torch
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats
import lerobot.common.datasets.lerobot_dataset as _lerobot_dataset
from lerobot.common.datasets.utils import check_delta_timestamps
from lerobot.common.datasets.utils import check_timestamps_sync
from lerobot.common.datasets.utils import get_delta_indices
from lerobot.common.datasets.utils import get_episode_data_index
from lerobot.common.datasets.utils import get_safe_version
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.video_utils import get_safe_default_codec

LeRobotDatasetMetadata = _lerobot_dataset.LeRobotDatasetMetadata
CODEBASE_VERSION = _lerobot_dataset.CODEBASE_VERSION


def _column_to_numpy(column) -> np.ndarray:
    """Convert an HF column to a numpy array without triggering image transforms."""
    array = np.asarray(column)
    if array.ndim > 1 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)
    return array


class LeRobotDataset(_lerobot_dataset.LeRobotDataset):
    """LeRobot dataset wrapper that delays image transforms until after timestamp validation."""

    def _load_hf_dataset(self, *, apply_transform: bool) -> datasets.Dataset:
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        if apply_transform:
            hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def load_hf_dataset(self) -> datasets.Dataset:
        return self._load_hf_dataset(apply_transform=True)

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        torch.utils.data.Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes retained for compatibility with upstream LeRobotDataset.
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.meta = LeRobotDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=force_cache_sync,
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self._load_hf_dataset(apply_transform=False)
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self._load_hf_dataset(apply_transform=False)

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(
            _column_to_numpy(self.hf_dataset["timestamp"]),
            _column_to_numpy(self.hf_dataset["episode_index"]),
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        # Enable the standard tensor conversion after the lightweight timestamp validation.
        self.hf_dataset.set_transform(hf_transform_to_torch)

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)
