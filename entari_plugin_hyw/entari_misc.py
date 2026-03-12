from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional

import httpx
from arclet.entari import Image, MessageChain
from PIL import Image as PILImage


def _compress_image_b64(b64_data: str, quality: int = 85) -> str:
    img_bytes = base64.b64decode(b64_data)
    img = PILImage.open(io.BytesIO(img_bytes))

    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(output.getvalue()).decode("utf-8")


async def _download_image(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def process_images(mc: MessageChain, vision_model: Optional[str] = None):
    if vision_model == "off":
        return [], None

    images: list[str] = []
    if mc.get(Image):
        urls = mc[Image].map(lambda item: item.src)
        raw_images = await asyncio.gather(*(_download_image(url) for url in urls))
        for image in raw_images:
            b64_raw = base64.b64encode(image).decode("utf-8")
            images.append(_compress_image_b64(b64_raw, quality=85))

    return images, None


__all__ = ["process_images"]
