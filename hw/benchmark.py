from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output.

    TODO:
        Handle cases like:
            "A"
            "(B)"
            "Answer: C"
            "The correct answer is D."
    """
    pattern = r"\b([" + "".join(choices) + r"])\b"
    match = re.search(pattern, text.strip())
    return match.group(1) if match else None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop.

    TODO:
        - load eval dataset;
        - build prompts;
        - call model.generate;
        - parse answers;
        - write predictions if output_path is provided;
        - return metrics.
    """
    from hw.dataset import MathVQADataset
    from hw.model import MathVLM, ModelConfig, SimpleTokenizer, make_tiny_backbones
    from hw.processor import MathVLMProcessor, ProcessorConfig

    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    inf_cfg = config.get("inference", {})

    manifest = data_cfg.get("eval_manifest") or data_cfg.get("train_manifest")
    split = data_cfg.get("split", "dev")
    max_samples = 4 if toy else data_cfg.get("max_samples")

    dataset = MathVQADataset(manifest, split=split, max_samples=max_samples)
    tokenizer = SimpleTokenizer()
    p_cfg = ProcessorConfig(
        image_size=proc_cfg.get("image_size", 224),
        num_tiles=proc_cfg.get("num_tiles", 1),
        num_image_tokens=proc_cfg.get("num_image_tokens", 16),
        max_length=proc_cfg.get("max_length", 256),
        ignore_index=proc_cfg.get("ignore_index", -100),
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
    model.eval()

    max_new_tokens = inf_cfg.get("max_new_tokens", 16)
    rows = []
    for i in range(len(dataset)):
        sample = dataset[i]
        prompt_text = processor.build_prompt(sample, include_answer=False)
        enc = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=p_cfg.max_length)
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long).unsqueeze(0)
        pixel_values = processor.preprocess_image(sample.image).unsqueeze(0)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

        gen_ids = model.generate(batch, max_new_tokens=max_new_tokens)
        gen_text = tokenizer.decode(gen_ids[0])
        prediction = parse_mc_answer(gen_text)

        rows.append({
            "id": sample.id,
            "answer": sample.answer,
            "prediction": prediction,
            "subject": sample.subject,
            "raw": gen_text,
        })

    output_path = inf_cfg.get("output_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
