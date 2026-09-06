from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class MultimodalEncoder(Protocol):
    """Maps text and images into the same vector space."""

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class ChineseClipEncoder:
    """Lazy optional adapter for OFA-Sys Chinese CLIP."""

    def __init__(self, model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16", device: str = "cpu"):
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        except ImportError as error:
            raise RuntimeError('中文 CLIP 依赖未安装，请执行 pip install -e ".[clip]"') from error
        self.torch = torch
        self.device = device
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model = ChineseCLIPModel.from_pretrained(model_name).to(device).eval()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vectors = self.model.get_text_features(**inputs)
        return normalize(vectors.float().cpu().numpy())

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        images = []
        try:
            for path in paths:
                images.append(Image.open(path).convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                vectors = self.model.get_image_features(**inputs)
            return normalize(vectors.float().cpu().numpy())
        finally:
            for image in images:
                image.close()

