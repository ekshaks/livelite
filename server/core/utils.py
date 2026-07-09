import asyncio
import reactivex as rx
from reactivex.subject import Subject as rx_Subject
from reactivex import Observable as rx_Observable, interval as rx_interval
from reactivex import operators as rx_ops
from reactivex.scheduler.eventloop import AsyncIOScheduler

import time

import time
import functools
import asyncio

def timeit(name: str = ""):
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                runtime = end_time - start_time
                print(f"{name or func.__name__} took {runtime:.4f} seconds to complete")
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                runtime = end_time - start_time
                print(f"{name or func.__name__} took {runtime:.4f} seconds to complete")
                return result
            return sync_wrapper
    return decorator

    
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

class Memory:
    def __init__(self):
        from agno.memory.v2.memory import Memory as AgnoMemory
        from agno.memory.v2.db.sqlite import SqliteMemoryDb
        from agno.models.message import Message

        self.Message = Message
        memory_db = SqliteMemoryDb(table_name="user_memories", db_file="tmp/agent.db")
        self.memory = AgnoMemory(
            db=memory_db
        )
    
    def add(self, content, role, user_id="default", session_id=None):
        self.memory.create_user_memories(
            messages=[
                self.Message(role=role, content=content),
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
    

        
