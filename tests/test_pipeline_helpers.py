import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import reactivex
from reactivex.subject import Subject

from server.core.audio_output import AudioChunk
from server.core.pipeline_helpers import add_text_sinks, add_tts
from server.core.stream_dsl import Stream, SubGroup
from server.core.tts_providers.piper import PiperTTSProvider


class FakeAudioOutput:
    async def write(self, chunk: AudioChunk) -> None:
        del chunk

    def clear(self) -> None:
        pass


class PipelineTextHelperTests(unittest.TestCase):
    def test_sensitive_transcript_can_skip_server_log_without_skipping_client(self):
        session = SimpleNamespace(send_to_client=Mock())
        subs = SubGroup()
        stream = Stream.source(reactivex.just("private verification claims"))

        with patch("server.core.pipeline_helpers.log_text_block") as log_text:
            add_text_sinks(
                stream,
                session,
                role="user",
                subs=subs,
                log=False,
            )

        log_text.assert_not_called()
        sent = session.send_to_client.call_args.args[0]
        self.assertEqual(sent.content, "private verification claims")
        subs.dispose()


class PipelineTTSHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_tts_returns_provider_for_session_lifecycle(self):
        subs = SubGroup()

        provider = add_tts(
            Stream.source(Subject()),
            FakeAudioOutput(),
            Stream.source(Subject()),
            subs=subs,
            mode="browser",
            provider="piper",
        )

        self.assertIsInstance(provider, PiperTTSProvider)
        subs.dispose()
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
