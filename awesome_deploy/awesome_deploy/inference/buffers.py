from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BufferState:
    values: dict[str, Any] = field(default_factory=dict)


class BufferManager:
    def __init__(self) -> None:
        self.state = BufferState()

    def reset(self) -> None:
        self.state.values.clear()

    def get(self, name: str, default=None):
        return self.state.values.get(name, default)

    def set(self, name: str, value) -> None:
        self.state.values[name] = value

    def update(self, updates: dict[str, Any]) -> None:
        self.state.values.update(updates)
