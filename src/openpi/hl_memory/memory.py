from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Sequence


def build_recent_context_indices(
    *,
    timestep: int,
    frame_subsample: int,
    recent_frames_length: int,
) -> list[int]:
    if timestep < 0:
        raise ValueError("timestep must be non-negative.")
    indices = list(range(timestep, -1, -frame_subsample))
    indices = list(reversed(indices[:recent_frames_length]))
    return indices


def map_relative_positions_to_absolute(
    relative_positions: Sequence[int],
    context_indices: Sequence[int],
) -> tuple[list[int], list[int]]:
    mapped: list[int] = []
    invalid: list[int] = []
    for position in relative_positions:
        if position <= 0 or position > len(context_indices):
            invalid.append(position)
            continue
        mapped.append(int(context_indices[position - 1]))
    return mapped, invalid


def cluster_candidate_indices(candidate_indices: Sequence[int], merge_distance: int) -> list[int]:
    if merge_distance < 0:
        raise ValueError("merge_distance must be non-negative.")
    if not candidate_indices:
        return []

    normalized = sorted(int(index) for index in candidate_indices)
    clusters: list[list[int]] = [[normalized[0]]]
    for index in normalized[1:]:
        if index - clusters[-1][-1] <= merge_distance:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return [int(statistics.median_low(cluster)) for cluster in clusters]


@dataclasses.dataclass
class EpisodicKeyframeMemory:
    memory_length: int
    merge_distance: int
    _candidate_indices: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if self.memory_length <= 0:
            raise ValueError("memory_length must be positive.")
        if self.merge_distance < 0:
            raise ValueError("merge_distance must be non-negative.")

    def add_candidates(self, absolute_indices: Sequence[int]) -> None:
        self._candidate_indices.extend(int(index) for index in absolute_indices)

    def candidate_indices(self) -> list[int]:
        return list(self._candidate_indices)

    def selected_indices(self) -> list[int]:
        clustered = cluster_candidate_indices(self._candidate_indices, self.merge_distance)
        return clustered[-self.memory_length :]

    def visible_indices(self, current_context_indices: Sequence[int]) -> list[int]:
        hidden = set(int(index) for index in current_context_indices)
        return [index for index in self.selected_indices() if index not in hidden]

