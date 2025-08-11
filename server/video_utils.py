import cv2
import numpy as np
import base64

def make_dummy_jpeg_base64(width=640, height=480, color=(0, 255, 0)):
    """
    Create a dummy JPEG image as a base64-encoded string.
    Default is a green image of size 640x480.
    """
    # Create a solid color image (BGR)
    img = np.full((height, width, 3), color, dtype=np.uint8)

    # Optionally, add text or timestamp
    cv2.putText(img, "Dummy Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

    # Encode to JPEG
    success, jpeg_bytes = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError("Failed to encode image to JPEG")

    # Convert to base64
    jpeg_b64 = base64.b64encode(jpeg_bytes)
    return jpeg_b64

async def feed_video(observer):
    await asyncio.sleep(1)  # let processor start pulling
    while True:
        observer.on_next(make_dummy_jpeg_base64())
        await asyncio.sleep(1)