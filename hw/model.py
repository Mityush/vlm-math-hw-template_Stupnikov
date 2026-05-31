from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from hw.constants import IGNORE_INDEX


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        # TODO: replace with a small projection network.
        # Recommended: LayerNorm -> Linear -> GELU -> Linear.
        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj_in = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.proj_out = nn.Linear(text_hidden_size, text_hidden_size)
        self.pool = nn.AdaptiveAvgPool1d(num_image_tokens)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.norm(vision_hidden_states)
        x = self.act(self.proj_in(x))
        x = self.proj_out(x)  # [B, seq, text_hidden]
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)  # [B, num_image_tokens, text_hidden]
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    result = input_embeds.clone()
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=False).squeeze(1)
        if positions.numel() > 0:
            result[b, positions] = visual_embeds[b, : positions.numel()]
    return result


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

        vision_out = self.vision_encoder(pixel_values)
        visual_embeds = self.adapter(vision_out)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)
        return self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        vision_out = self.vision_encoder(pixel_values)
        visual_embeds = self.adapter(vision_out)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs
        )
