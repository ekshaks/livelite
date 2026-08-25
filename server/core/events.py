from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class SpeechEvent(Enum):
    SPEECH_START = auto()
    SPEECH_END = auto()

    def __str__(self):
        return self.name


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    is_final: bool
    utterance_id: str | None = None
    context: Any = None


@dataclass(frozen=True)
class ClientTranscriptMessage:
    role: str
    content: str

    def to_client_dict(self) -> dict[str, Any]:
        return {
            "type": "transcript",
            "role": self.role,
            "content": self.content,
        }
