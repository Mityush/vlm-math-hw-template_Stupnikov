from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert("RGB")
        cfg = self.config
        grid = math.ceil(math.sqrt(cfg.num_tiles))
        scaled = image.resize((cfg.image_size * grid, cfg.image_size * grid), Image.BICUBIC)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tiles = []
        for row in range(grid):
            for col in range(grid):
                if len(tiles) >= cfg.num_tiles:
                    break
                x0, y0 = col * cfg.image_size, row * cfg.image_size
                tile = scaled.crop((x0, y0, x0 + cfg.image_size, y0 + cfg.image_size))
                arr = np.array(tile, dtype=np.float32) / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
                tiles.append((t - mean) / std)
        return torch.stack(tiles)  # [num_tiles, 3, H, W]

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        img_str = f"{IMAGE_START_TOKEN} {image_tokens} {IMAGE_END_TOKEN}"
        options_text = "\n".join(sample.options)
        prompt = f"{img_str}\nВопрос: {sample.question}\nВарианты:\n{options_text}\nОтвет:"
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        cfg = self.config
        prompt_prefix = self.build_prompt(sample, include_answer=False)
        prompt_full = self.build_prompt(sample, include_answer=True)
        enc_prefix = self.tokenizer(prompt_prefix, add_special_tokens=False, truncation=True, max_length=cfg.max_length)
        enc_full = self.tokenizer(prompt_full, add_special_tokens=True, truncation=True, max_length=cfg.max_length)
        input_ids = torch.tensor(enc_full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc_full["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        prefix_len = min(len(enc_prefix["input_ids"]), len(input_ids))
        labels[:prefix_len] = cfg.ignore_index
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        max_len = max(item["input_ids"].shape[0] for item in batch)
        pad_id = self.tokenizer.pad_token_id
        ignore = self.config.ignore_index
        input_ids_out, attn_out, labels_out, pv_out = [], [], [], []
        for item in batch:
            L = item["input_ids"].shape[0]
            pad = max_len - L
            input_ids_out.append(torch.cat([item["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            attn_out.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels_out.append(torch.cat([item["labels"], torch.full((pad,), ignore, dtype=torch.long)]))
            pv_out.append(item["pixel_values"])
        return {
            "input_ids": torch.stack(input_ids_out),
            "attention_mask": torch.stack(attn_out),
            "labels": torch.stack(labels_out),
            "pixel_values": torch.stack(pv_out),
        }
