"""Bounded, ordered PCM16 delivery for WebSocket voice responses."""

from __future__ import annotations

import asyncio

from .audio_output import AudioChunk


class WebSocketPCMOutput:
    def __init__(self, send_bytes, send_lock: asyncio.Lock, *, max_chunks: int = 64) -> None:
        self._send_bytes = send_bytes
        self._send_lock = send_lock
        self._queue: asyncio.Queue[tuple[int, bytes, int]] = asyncio.Queue(maxsize=max_chunks)
        self._generation = 0
        self._worker = asyncio.create_task(self._send_loop(), name="voice-websocket-pcm")

    async def write(self, chunk: AudioChunk) -> None:
        if chunk.channels != 1:
            raise ValueError("WebSocket PCM output requires mono chunks")
        generation = self._generation
        await self._queue.put((generation, chunk.samples.astype("<i2", copy=False).tobytes(), chunk.sample_rate))

    def clear(self) -> None:
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                return

    async def wait_until_drained(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        self.clear()
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass

    async def _send_loop(self) -> None:
        while True:
            generation, payload, sample_rate = await self._queue.get()
            try:
                if generation == self._generation:
                    async with self._send_lock:
                        if generation == self._generation:
                            await self._send_bytes(payload)
                    await asyncio.sleep((len(payload) / 2) / sample_rate)
            finally:
                self._queue.task_done()
