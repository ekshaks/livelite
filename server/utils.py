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



class Memory:
    from agno.memory.v2.memory import Memory
    from agno.memory.v2.db.sqlite import SqliteMemoryDb
    from agno.models.message import Message


    def __init__(self):
        memory_db = SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db")
        self.memory = Memory(
            db=memory_db
        )
    
    def add(self, content, role, user_id="default", session_id=None):
        self.memory.create_user_memories(
            messages=[
                Message(role=role, content=content),
            ],
            user_id=user_id,
            session_id=session_id
        )
        return
        
        self.memory.append(
            dict(
                user_id=user_id,
                session_id=session_id,
                content=content
            )
        )
    def add_session_memory(self, content, session_id):
        self.memory.add_session_memory(
            session_id=session_id,
            content=content
        )
        
    
    def get_user_memories(self, user_id):
        return self.memory.get_user_memories(user_id=user_id)
    
    def get_session_memories(self, session_id):
        return self.memory.get_session_memories(session_id=session_id)
    

        
