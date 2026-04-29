from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import json

_PREDICTION_REQUIRED_KEYS = {
    "updated_language_memory",
    "current_subtask",
    "phase",
    "target_query",
    "goal_query",
}


@dataclasses.dataclass(frozen=True)
class HLMemoryPrediction:
    updated_language_memory: str
    current_subtask: str
    keyframe_candidate_positions: tuple[int, ...]
    phase: str
    target_query: str
    goal_query: str

    def __post_init__(self) -> None:
        if not self.updated_language_memory.strip():
            raise ValueError("updated_language_memory must be non-empty.")
        if not self.current_subtask.strip():
            raise ValueError("current_subtask must be non-empty.")
        if not self.phase.strip():
            raise ValueError("phase must be non-empty.")
        for position in self.keyframe_candidate_positions:
            if position <= 0:
                raise ValueError("keyframe_candidate_positions must be positive and 1-indexed.")

    def to_dict(self) -> dict[str, object]:
        return {
            "updated_language_memory": self.updated_language_memory,
            "current_subtask": self.current_subtask,
            "keyframe_candidate_positions": list(self.keyframe_candidate_positions),
            "phase": self.phase,
            "target_query": self.target_query,
            "goal_query": self.goal_query,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))

    def with_recent_position_limit(self, recent_valid_length: int) -> "HLMemoryPrediction":
        """Drops keyframe positions that point outside the valid recent clip."""
        limit = max(int(recent_valid_length), 0)
        return dataclasses.replace(
            self,
            keyframe_candidate_positions=tuple(
                position for position in self.keyframe_candidate_positions if 1 <= position <= limit
            ),
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HLMemoryPrediction":
        missing = _PREDICTION_REQUIRED_KEYS - set(data)
        if missing:
            raise ValueError(f"Missing prediction keys: {sorted(missing)}")
        raw_positions = data.get("keyframe_candidate_positions", data.get("keyframe_positions", []))
        if not isinstance(raw_positions, list):
            raise ValueError("keyframe_candidate_positions must be a list.")
        keyframe_candidate_positions: list[int] = []
        for raw_position in raw_positions:
            try:
                position = int(raw_position)
            except (TypeError, ValueError):
                continue
            if position <= 0:
                continue
            if position not in keyframe_candidate_positions:
                keyframe_candidate_positions.append(position)
        return cls(
            updated_language_memory=str(data["updated_language_memory"]).strip(),
            current_subtask=str(data["current_subtask"]).strip(),
            keyframe_candidate_positions=tuple(keyframe_candidate_positions),
            phase=str(data["phase"]).strip(),
            target_query=str(data["target_query"]).strip(),
            goal_query=str(data["goal_query"]).strip(),
        )

    @classmethod
    def from_json(cls, text: str) -> "HLMemoryPrediction":
        return cls.from_dict(_extract_first_json_object(text))


def _extract_first_json_object(text: str) -> dict[str, object]:
    fallback: dict[str, object] | None = None
    for candidate_text in _candidate_json_texts(text):
        parsed_objects = list(_iter_json_objects(candidate_text))
        prediction_objects = [parsed for parsed in parsed_objects if _looks_like_prediction(parsed)]
        if prediction_objects:
            return prediction_objects[-1]
        if fallback is None and parsed_objects:
            fallback = parsed_objects[-1]

    if fallback is not None:
        return fallback

    raise ValueError("Could not parse a JSON object from model output.")


def _candidate_json_texts(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []
    if "</think>" in stripped:
        candidates.append(stripped.rsplit("</think>", maxsplit=1)[-1].strip())
    candidates.extend(_extract_fenced_blocks(stripped))
    candidates.append(_strip_fenced_block(stripped))
    candidates.append(stripped)

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _iter_json_objects(text: str) -> Iterator[dict[str, object]]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def _looks_like_prediction(data: dict[str, object]) -> bool:
    return _PREDICTION_REQUIRED_KEYS.issubset(data)


def _extract_fenced_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    lines = text.splitlines()
    in_block = False
    current: list[str] = []
    for line in lines:
        if line.startswith("```"):
            if in_block:
                blocks.append("\n".join(current).strip())
                current = []
                in_block = False
            else:
                in_block = True
                current = []
            continue
        if in_block:
            current.append(line)
    return blocks


def _strip_fenced_block(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text
