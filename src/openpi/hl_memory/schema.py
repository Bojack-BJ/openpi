from __future__ import annotations

import dataclasses
import json


@dataclasses.dataclass(frozen=True)
class HLMemoryPrediction:
    updated_language_memory: str
    current_subtask: str
    keyframe_positions: tuple[int, ...]
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
        for position in self.keyframe_positions:
            if position <= 0:
                raise ValueError("keyframe_positions must be positive and 1-indexed.")

    def to_dict(self) -> dict[str, object]:
        return {
            "updated_language_memory": self.updated_language_memory,
            "current_subtask": self.current_subtask,
            "keyframe_positions": list(self.keyframe_positions),
            "phase": self.phase,
            "target_query": self.target_query,
            "goal_query": self.goal_query,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "HLMemoryPrediction":
        required_keys = {
            "updated_language_memory",
            "current_subtask",
            "keyframe_positions",
            "phase",
            "target_query",
            "goal_query",
        }
        missing = required_keys - set(data)
        if missing:
            raise ValueError(f"Missing prediction keys: {sorted(missing)}")
        raw_positions = data["keyframe_positions"]
        if not isinstance(raw_positions, list):
            raise ValueError("keyframe_positions must be a list.")
        keyframe_positions: list[int] = []
        for raw_position in raw_positions:
            position = int(raw_position)
            if position not in keyframe_positions:
                keyframe_positions.append(position)
        return cls(
            updated_language_memory=str(data["updated_language_memory"]).strip(),
            current_subtask=str(data["current_subtask"]).strip(),
            keyframe_positions=tuple(keyframe_positions),
            phase=str(data["phase"]).strip(),
            target_query=str(data["target_query"]).strip(),
            goal_query=str(data["goal_query"]).strip(),
        )

    @classmethod
    def from_json(cls, text: str) -> "HLMemoryPrediction":
        return cls.from_dict(_extract_first_json_object(text))


def _extract_first_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_fenced_block(stripped)

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from model output.")


def _strip_fenced_block(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text
