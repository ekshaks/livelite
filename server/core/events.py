from enum import Enum, auto

class SpeechEvent(Enum):
    SPEECH_START = auto()
    SPEECH_END = auto()

    def __str__(self):
        return self.name