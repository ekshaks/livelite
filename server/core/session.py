import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict

from aiortc import RTCPeerConnection


@dataclass
class SessionContext:
    """Runtime resources and lifecycle signals for one WebRTC session."""

    pc: RTCPeerConnection
    data_channels: Dict[str, Any]
    audio_input: Any
    video_input: Any
    client_input: Any
    main_loop: asyncio.AbstractEventLoop
    user_id: str | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    closed: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait_until_ready(self) -> None:
        await self.ready.wait()

    @property
    def assistant_audio_track(self):
        return getattr(self.pc, "assistant_audio_track", None)
