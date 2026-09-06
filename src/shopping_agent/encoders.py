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

    def __init__(
        self,
        model_name: str = "OFA-Sys/chinese-clip-vit-base-patch16",
        device: str = "cpu",
        batch_size: int = 16,
    ):
        try:
            import torch
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        except ImportError as error:
            raise RuntimeError('中文 CLIP 依赖未安装，请执行 pip install -e ".[clip]"') from error
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
        self.model = ChineseCLIPModel.from_pretrained(model_name).to(device).eval()

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        batches = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.processor(
                text=list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                # Some published Chinese-CLIP configs disable the BERT pooler,
                # while recent Transformers get_text_features expects it.
                # Chinese-CLIP uses the first token as its text representation.
                outputs = self.model.text_model(**inputs)
                vectors = self.model.text_projection(outputs.last_hidden_state[:, 0, :])
            batches.append(vectors.float().cpu().numpy())
        return normalize(np.concatenate(batches))

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        from PIL import Image

        batches = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            try:
                for path in paths[start : start + self.batch_size]:
                    images.append(Image.open(path).convert("RGB"))
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.inference_mode():
                    vectors = self.model.get_image_features(**inputs)
                batches.append(vectors.float().cpu().numpy())
            finally:
                for image in images:
                    image.close()
        return normalize(np.concatenate(batches))
