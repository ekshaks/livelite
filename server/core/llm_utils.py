from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import base64
import io

async def vlm_agent(text, last_frame, PROMPTS_FILE, system_prompt_id='vlm_math'):
    from PIL import Image
    import google.generativeai as genai

    with open(PROMPTS_FILE) as f:
        system_prompt = yaml.safe_load(f)[system_prompt_id]

    model = genai.GenerativeModel('gemini-2.0-flash')

    # Create the parts of the prompt
    prompt_parts = [
        system_prompt,
        text,
    ]
    if last_frame is not None:
        pil_image = Image.fromarray(last_frame.to_ndarray(format='rgb24'))
        prompt_parts.append(pil_image)
    print('call vlm agent', prompt_parts)
    
    # Send the request to the API
    response = model.generate_content(prompt_parts)
    print(response.text)
    return response



def numpy_to_base64(np_array, format="PNG"):
    # Convert to PIL Image
    pil_img = PILImage.fromarray(np_array.astype("uint8"))
    # Save to BytesIO
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    buffer.seek(0)
    # Encode to base64
    return base64.b64encode(buffer.read()).decode("utf-8")

async def call_vlm_agent(agent, text, last_frame):
    from agno.media import Image as AgnoImage

    # agent = Agent(
    # model=Gemini(id="gemini-2.0-flash"),
    # markdown=True,
    # )

    print('last frame ', last_frame)
    last_frame: 'av.video.frame.VideoFrame'
    if last_frame is not None:
        buffer = io.BytesIO()
        PILImage.fromarray(last_frame.to_ndarray(format='rgb24')).save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        # image_bytes = PILImage.fromarray(last_frame.to_ndarray(format='rgb24')).tobytes("png")
        image = AgnoImage(content=image_bytes)
        images = [image]
    else:
        images = []
    
    response = await agent.arun(
        text,
        images=images,
        #stream=True,
    )
    print(response.content)

    return response