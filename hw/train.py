from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss.

    TODO:
        - model.train();
        - forward;
        - ensure finite loss;
        - backward;
        - optimizer.step();
        - optimizer.zero_grad();
    """
    model.train()
    output = model(batch)
    loss = output["loss"] if isinstance(output, dict) else output
    assert loss.isfinite(), f"Non-finite loss: {loss}"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point.

    TODO:
        - instantiate dataset, processor, model;
        - create DataLoader;
        - support max_steps and fast_train;
        - save adapter/checkpoint if configured.
    """
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig, SimpleTokenizer, make_tiny_backbones
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    proc_cfg = config.get("processor", {})
    trainer_cfg = config.get("trainer", {})

    tokenizer = SimpleTokenizer()
    p_cfg = ProcessorConfig(
        image_size=proc_cfg.get("image_size", 224),
        num_tiles=proc_cfg.get("num_tiles", 1),
        tile_overlap=proc_cfg.get("tile_overlap", 0.0),
        num_image_tokens=proc_cfg.get("num_image_tokens", 16),
        max_length=proc_cfg.get("max_length", 256),
        ignore_index=proc_cfg.get("ignore_index", -100),
    )
    dataset = MathVQADataset(
        data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )
    processor = MathVLMProcessor(tokenizer, p_cfg)

    vision_encoder, language_model = make_tiny_backbones()
    model_config = ModelConfig(
        vision_hidden_size=vision_encoder.hidden_size,
        text_hidden_size=language_model.hidden_size,
        num_image_tokens=p_cfg.num_image_tokens,
        image_token_id=tokenizer.vocab.get("<image>", 2),
    )
    model = MathVLM(vision_encoder, language_model, model_config)
    if model_cfg.get("freeze_vision", True) or model_cfg.get("freeze_llm", True):
        model.freeze_backbones()

    local_bs = trainer_cfg.get("local_batch_size", 1)
    global_bs = trainer_cfg.get("global_batch_size", local_bs)
    grad_accum = max(1, global_bs // max(local_bs, 1))
    max_steps = 1 if fast_train else trainer_cfg.get("max_steps")
    num_epochs = trainer_cfg.get("num_train_epochs", 1)

    class _Processed(torch.utils.data.Dataset):
        def __init__(self, ds, proc):
            self.ds, self.proc = ds, proc
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return self.proc(self.ds[idx])

    loader = torch.utils.data.DataLoader(
        _Processed(dataset, processor),
        batch_size=local_bs,
        collate_fn=processor.collate,
        num_workers=trainer_cfg.get("num_workers", 0),
        shuffle=True,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=trainer_cfg.get("learning_rate", 5e-4),
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
    )

    step = 0
    accum_count = 0
    optimizer.zero_grad()
    done = False
    for _epoch in range(num_epochs):
        for batch in loader:
            model.train()
            output = model(batch)
            loss = output["loss"] if isinstance(output, dict) else output
            assert loss.isfinite(), f"Non-finite loss: {loss}"
            (loss / grad_accum).backward()
            accum_count += 1
            if accum_count % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                print(f"step={step} loss={loss.item():.4f}")
                if max_steps is not None and step >= max_steps:
                    done = True
                    break
        if done:
            break

    ckpt = trainer_cfg.get("save_checkpoint_path")
    if ckpt:
        Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.adapter.state_dict(), ckpt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
