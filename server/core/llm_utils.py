import base64
import asyncio
import io
import time

import yaml
from PIL import Image as PILImage

from .logging_utils import log_text_block, monitor_log, monitor_time
from .utils import timeit


def create_agent(model_id, prompts_path, prompt_id, name="Agent"):
    from agno.agent import Agent
    from agno.models.google import Gemini
    from agno.models.groq import Groq

    with open(prompts_path) as f:
        prompts = yaml.safe_load(f) or {}
    prompt = prompts[prompt_id]

    if "google" in model_id:
        model = Gemini(id=model_id.split(":")[-1])
    elif "groq" in model_id:
        model = Groq(id=model_id.split(":")[-1])
    else:
        raise ValueError(f"Unknown model: {model_id}")

    return Agent(
        name=name,
        description=prompt["description"],
        instructions=[prompt["instructions"]],
        model=model,
        markdown=True,
    )


def split_spoken_written(text):
    text = (text or "").strip()
    upper = text.upper()
    if "SPOKEN:" not in upper and "WRITTEN:" not in upper:
        return {"spoken": text, "written": text}

    def section(label, stop_label):
        start = upper.find(label)
        if start == -1:
            return ""
        start += len(label)
        end = upper.find(stop_label, start)
        return text[start : end if end != -1 else None].strip()

    spoken = section("SPOKEN:", "WRITTEN:")
    written = section("WRITTEN:", "SPOKEN:")
    spoken = spoken or written
    written = written or spoken
    return {"spoken": spoken, "written": written}


def numpy_to_base64(np_array, format="PNG"):
    pil_img = PILImage.fromarray(np_array.astype("uint8"))
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def groq_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[1] if model_id.startswith("groq:") else model_id


def groq_reasoning_effort(model_id: str, requested: str | None) -> str | None:
    if requested != "none":
        return requested
    model = groq_model_id(model_id)
    if model.startswith("openai/") or "gpt-oss" in model:
        return "low"
    return requested


def gemini_model_id(model_id: str) -> str:
    return model_id.split(":", 1)[1] if model_id.startswith("gemini:") else model_id


def video_frame_to_png_bytes(frame) -> bytes:
    buffer = io.BytesIO()
    PILImage.fromarray(frame.to_ndarray(format="rgb24")).save(buffer, format="PNG")
    return buffer.getvalue()


def video_frame_to_data_url(frame, format="PNG") -> str:
    if format != "PNG":
        buffer = io.BytesIO()
        PILImage.fromarray(frame.to_ndarray(format="rgb24")).save(buffer, format=format)
        image_bytes = buffer.getvalue()
    else:
        image_bytes = video_frame_to_png_bytes(frame)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def log_groq_error(exc: Exception):
    body = getattr(exc, "body", None)
    failed_generation = None
    if isinstance(body, dict):
        error = body.get("error") or body
        failed_generation = error.get("failed_generation")
    monitor_log(f"groq api error: {type(exc).__name__}: {exc}")
    if failed_generation is not None:
        log_text_block(
            "GROQ FAILED GENERATION",
            failed_generation,
            max_chars=10000,
        )


async def call_groq_chat(
    model_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    reasoning_effort: str | None = "none",
    temperature: float = 0,
    max_completion_tokens: int = 1024,
) -> str:
    from groq import AsyncGroq

    client = AsyncGroq()
    model = groq_model_id(model_id)
    started_at = time.perf_counter()
    monitor_log(f"llm request start provider=groq operation=chat model={model}")
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            reasoning_effort=groq_reasoning_effort(model_id, reasoning_effort),
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started_at
        monitor_time(
            "llm",
            "chat",
            elapsed_s,
            provider="groq",
            model=model,
            outcome="failed",
            error=type(exc).__name__,
        )
        log_groq_error(exc)
        raise
    elapsed_s = time.perf_counter() - started_at
    monitor_time("llm", "chat", elapsed_s, provider="groq", model=model)
    content = completion.choices[0].message.content or ""
    log_text_block("RAW GROQ RESPONSE", content, max_chars=10000)
    return content


async def call_groq_vision(
    model_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    frame,
    response_format: dict | None = None,
    reasoning_effort: str | None = "none",
    temperature: float = 0,
    max_completion_tokens: int = 1024,
) -> str:
    if frame is None:
        raise ValueError("Camera frame unavailable")

    from groq import AsyncGroq

    client = AsyncGroq()
    model = groq_model_id(model_id)
    started_at = time.perf_counter()
    monitor_log(f"llm request start provider=groq operation=vision model={model}")
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": video_frame_to_data_url(frame)},
                        },
                    ],
                },
            ],
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            reasoning_effort=groq_reasoning_effort(model_id, reasoning_effort),
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started_at
        monitor_time(
            "llm",
            "vision",
            elapsed_s,
            provider="groq",
            model=model,
            outcome="failed",
            error=type(exc).__name__,
        )
        log_groq_error(exc)
        raise
    elapsed_s = time.perf_counter() - started_at
    monitor_time("llm", "vision", elapsed_s, provider="groq", model=model)
    content = completion.choices[0].message.content or ""
    log_text_block("RAW GROQ VISION RESPONSE", content, max_chars=10000)
    return content


def call_gemini_vision_sync(
    model_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    frame,
    response_mime_type: str | None = None,
    thinking_level: str | None = None,
    max_output_tokens: int = 8192,
) -> str:
    if frame is None:
        raise ValueError("Camera frame unavailable")

    monitor_log(f"gemini vision start model={model_id}")
    from google import genai
    from google.genai import types

    monitor_log("gemini vision creating client")
    client = genai.Client()
    monitor_log("gemini vision client created")
    config = {
        "system_instruction": system_prompt,
        "max_output_tokens": max_output_tokens,
    }
    if response_mime_type is not None:
        config["response_mime_type"] = response_mime_type
    if thinking_level is not None:
        config["thinking_config"] = {"thinking_level": thinking_level}

    monitor_log("gemini vision converting frame to png bytes")
    image_bytes = video_frame_to_png_bytes(frame)
    monitor_log(f"gemini vision frame converted bytes={len(image_bytes)}")
    monitor_log("gemini vision generate_content request start")
    started_at = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=gemini_model_id(model_id),
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                user_prompt,
            ],
            config=config,
        )
    except Exception as exc:
        monitor_time(
            "llm",
            "vision",
            time.perf_counter() - started_at,
            provider="gemini",
            model=gemini_model_id(model_id),
            outcome="failed",
            error=type(exc).__name__,
        )
        raise
    monitor_log("gemini vision generate_content request done")
    monitor_time(
        "llm",
        "vision",
        time.perf_counter() - started_at,
        provider="gemini",
        model=gemini_model_id(model_id),
    )
    content = response.text or ""
    log_text_block("RAW GEMINI VISION RESPONSE", content, max_chars=10000)
    return content


async def call_gemini_vision(
    model_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    frame,
    response_mime_type: str | None = None,
    thinking_level: str | None = None,
    max_output_tokens: int = 8192,
) -> str:
    return await asyncio.to_thread(
        call_gemini_vision_sync,
        model_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        frame=frame,
        response_mime_type=response_mime_type,
        thinking_level=thinking_level,
        max_output_tokens=max_output_tokens,
    )


async def call_vision_model(
    model_id: str,
    *,
    system_prompt: str,
    user_prompt: str,
    frame,
    json_mode: bool = False,
    max_output_tokens: int = 512,
) -> str:
    if model_id.startswith("gemini:") or model_id.startswith("models/gemini"):
        return await call_gemini_vision(
            model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frame=frame,
            response_mime_type="application/json" if json_mode else None,
            max_output_tokens=max_output_tokens,
        )
    if model_id.startswith("groq:"):
        return await call_groq_vision(
            model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            frame=frame,
            response_format={"type": "json_object"} if json_mode else None,
            max_completion_tokens=max_output_tokens,
        )
    raise ValueError(f"Unknown vision model provider: {model_id}")


async def vlm_agent(text, last_frame, prompts_file, system_prompt_id="vlm_math"):
    from PIL import Image
    import google.generativeai as genai

    with open(prompts_file) as f:
        system_prompt = yaml.safe_load(f)[system_prompt_id]

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt_parts = [system_prompt, text]
    if last_frame is not None:
        pil_image = Image.fromarray(last_frame.to_ndarray(format="rgb24"))
        prompt_parts.append(pil_image)
    monitor_log(
        f"call_vlm_agent prompt_parts={len(prompt_parts)} "
        f"has_image={last_frame is not None}"
    )

    started_at = time.perf_counter()
    response = model.generate_content(prompt_parts)
    monitor_time("llm", "vision", time.perf_counter() - started_at, provider="gemini")
    log_text_block("RAW VLM RESPONSE", response.text, max_chars=10000)
    return response


async def call_vlm_agent(agent, text, last_frame):
    from agno.media import Image as AgnoImage

    monitor_log(f"call_vlm_agent last_frame_present={last_frame is not None}")
    last_frame: "av.video.frame.VideoFrame"
    if last_frame is not None:
        buffer = io.BytesIO()
        PILImage.fromarray(last_frame.to_ndarray(format="rgb24")).save(
            buffer, format="PNG"
        )
        image = AgnoImage(content=buffer.getvalue())
        images = [image]
    else:
        images = []

    monitor_log(f"call_vlm_agent images_count={len(images)}")

    started_at = time.perf_counter()
    response = await agent.arun(text, images=images)
    monitor_time("llm", "agent_run", time.perf_counter() - started_at, provider="agent")
    log_text_block("RAW VLM RESPONSE", response.content, max_chars=10000)
    return response


@timeit(name="call", service="llm")
async def call_llm(agent, text, last_frame, mode="av"):
    if mode == "av":
        response = await call_vlm_agent(agent, text, last_frame)
        return response.content
    response = await agent.arun(text)
    return response.content
