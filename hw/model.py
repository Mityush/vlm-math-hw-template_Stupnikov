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
    
_TINY_VIS_HIDDEN = 32
_TINY_TXT_HIDDEN = 64
_TINY_VOCAB = 4096
_TINY_PATCHES = 9

class _TinyVisionEncoder(nn.Module):
    hidden_size: int = _TINY_VIS_HIDDEN

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, _TINY_VIS_HIDDEN)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = pixel_values.shape
        x = pixel_values.reshape(B * T, C, H * W)
        x = F.adaptive_avg_pool1d(x, _TINY_PATCHES).transpose(1, 2)  # [BT, P, 3]
        x = self.proj(x)  # [BT, P, hidden]
        return x.reshape(B, T * _TINY_PATCHES, _TINY_VIS_HIDDEN)

class _TinyLanguageModel(nn.Module):
    hidden_size: int = _TINY_TXT_HIDDEN

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(_TINY_VOCAB, _TINY_TXT_HIDDEN)
        self.lm_head = nn.Linear(_TINY_TXT_HIDDEN, _TINY_VOCAB)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        logits = self.lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().reshape(-1, _TINY_VOCAB)
            shift_labels = labels[:, 1:].contiguous().reshape(-1).clamp(min=-100)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 16,
        **kwargs: Any,
    ) -> torch.Tensor:
        cur = inputs_embeds
        out: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            logits = self.lm_head(cur)
            nxt = logits[:, -1].argmax(dim=-1)  # [B]
            out.append(nxt)
            cur = torch.cat([cur, self.embed_tokens(nxt.unsqueeze(1))], dim=1)
        return torch.stack(out, dim=1)  # [B, max_new_tokens]

class SimpleTokenizer:
    """Minimal whitespace tokenizer for Track A smoke tests."""

    pad_token_id: int = 0
    eos_token_id: int = 1

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1, "<image>": 2, "<image_start>": 3, "<image_end>": 4}
        self._next_id = 5

    def _id(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self._next_id
            self._next_id += 1
        return self.vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self._id(t) for t in text.replace("\n", " ").split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        rev = {v: k for k, v in self.vocab.items()}
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(rev.get(i, "") for i in ids)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

def make_tiny_backbones() -> tuple[_TinyVisionEncoder, _TinyLanguageModel]:
    """Return (vision_encoder, language_model) tiny mocked models for Track A."""
    return _TinyVisionEncoder(), _TinyLanguageModel()
