import time
import unittest

import numpy as np
from reactivex.subject import Subject

from server.core.events import SpeechEvent
from server.core.stream_dsl import Stream, SubGroup, turn_detector


class TurnDetectorTests(unittest.TestCase):
    def test_named_outputs_share_one_audio_subscription_and_dispose(self):
        audio = Subject()
        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
            silence_timeout=0.01,
            poll_interval=0.005,
        )
        segments = []
        signals = []
        subs = SubGroup()

        segment_sub = turn.segments.to(
            lambda observable: observable.subscribe(segments.append),
            subs=subs,
        )
        signal_sub = turn.signals.to(
            lambda observable: observable.subscribe(signals.append),
            subs=subs,
        )

        self.assertEqual(len(audio.observers), 1)
        audio.on_next(np.array([1, 2, 3], dtype=np.int16))
        time.sleep(0.04)

        self.assertEqual(signals, [SpeechEvent.SPEECH_START, SpeechEvent.SPEECH_END])
        self.assertEqual(len(segments), 1)
        np.testing.assert_array_equal(
            segments[0],
            np.array([1, 2, 3], dtype=np.int16),
        )

        segment_sub.dispose()
        self.assertEqual(len(audio.observers), 1)
        signal_sub.dispose()
        self.assertEqual(len(audio.observers), 0)

    def test_completion_flushes_buffered_speech(self):
        audio = Subject()
        turn = Stream.source(audio) | turn_detector(
            is_speech_fn=lambda _chunk: True,
            silence_timeout=10,
            poll_interval=1,
        )
        segments = []
        completed = []
        subs = SubGroup()

        turn.segments.to(
            lambda observable: observable.subscribe(
                on_next=segments.append,
                on_completed=lambda: completed.append(True),
            ),
            subs=subs,
        )

        audio.on_next(np.array([4, 5], dtype=np.int16))
        audio.on_completed()

        self.assertEqual(len(segments), 1)
        self.assertEqual(completed, [True])
        self.assertEqual(len(audio.observers), 0)
        subs.dispose()


if __name__ == "__main__":
    unittest.main()
