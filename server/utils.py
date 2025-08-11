import asyncio
import reactivex as rx
from reactivex.subject import Subject as rx_Subject
from reactivex import Observable as rx_Observable, interval as rx_interval
from reactivex import operators as rx_ops
from reactivex.scheduler.eventloop import AsyncIOScheduler

def rx_to_async_iter(rx_observable, debug=False):
    """Convert RxPY Observable to AsyncIterable of ProcessorPart."""
    queue = asyncio.Queue()

    def on_next(item):
        if debug: print('rx_to_async_iter', item)
        queue.put_nowait(item)
    def on_completed():
        queue.put_nowait(None)

    rx_observable.subscribe(on_next, on_error=queue.put_nowait, on_completed=on_completed,
                            scheduler=AsyncIOScheduler(asyncio.get_event_loop()))

    async def generator():
        while True:
            item = await queue.get()
            if item is None:
                break
            if debug: print('rx_to_async_iter yield', item)
            yield item

    return generator()



def send_text_to_client(text, data_channels, loop, role="user", channel="server_text"):
    import json
    #print(f"sending to client: {text}")
    try:
        channel = data_channels.get(channel)
        if channel and channel.readyState == "open":
            data = json.dumps({"role": role, "content": text})
            loop.call_soon_threadsafe(lambda: channel.send(data))
    except Exception as e:
        print(f"Error sending text to client: {e}")


