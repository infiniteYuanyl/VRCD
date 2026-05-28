#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

warnings.filterwarnings(
    "ignore",
    message=r"for vision_model\..*copying from a non-meta parameter.*",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEMOS_PATH = ROOT / "examples" / "demos.json"
MASK_TOKEN_ID = 126336
DEFAULT_MODEL = "hbXNov/lavida-llada-reason"
DEFAULT_DECODING_METHOD = "vrcd"
DEFAULT_IMG_PATH = ROOT / "examples" / "objects.png"


def parse_dtype(name: str) -> tuple[str, torch.dtype]:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return "bfloat16", torch.bfloat16
    if name in {"fp16", "float16"}:
        return "float16", torch.float16
    if name in {"fp32", "float32"}:
        return "float32", torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_vision_kwargs(vision_tower: str) -> dict[str, Any]:
    return {
        "mm_vision_tower": vision_tower,
        "mm_resampler_type": None,
        "mm_projector_type": os.getenv("LLADA_VISION_PROJECTOR", "mlp2x_gelu"),
        "mm_hidden_size": int(os.getenv("LLADA_VISION_ENCODER_HIDDEN_SIZE", "1152")),
        "mm_pooler_ratio": int(os.getenv("LLADA_MM_POOLER_RATIO", "2")),
        "use_mm_proj": True,
        "mm_patch_merge_type": "spatial_unpad",
    }


def disable_activation_checkpointing(model) -> None:
    modules = [model]
    if hasattr(model, "get_model"):
        modules.append(model.get_model())
    inner = getattr(model, "model", None)
    if inner is not None:
        modules.append(inner)
    for module in modules:
        setter = getattr(module, "set_activation_checkpointing", None)
        if setter is not None:
            setter(None)


def load_demos() -> list[dict[str, Any]]:
    with DEMOS_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["demos"])


def find_demo_by_image(image_path: Path) -> dict[str, Any] | None:
    resolved = image_path.resolve()
    for demo in load_demos():
        demo_image = Path(demo["image"])
        if not demo_image.is_absolute():
            demo_image = ROOT / demo_image
        if demo_image.resolve() == resolved:
            return demo
    return None


def has_text(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "none", "null"}


def format_demo_question(demo: dict[str, Any]) -> str:
    question_parts = []
    if has_text(demo.get("hint")):
        question_parts.append(str(demo["hint"]).strip())
    question_parts.append(str(demo["question"]).strip())
    lines = [" ".join(question_parts) + " There are several options:"]
    for letter, text in sorted(demo["choices"].items()):
        lines.append(f"{letter}. {text}")
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def build_prompt(question: str, conv_template: str, default_image_token: str, conv_templates) -> str:
    query = question if default_image_token in question else default_image_token + "\n" + question
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_image_tensor(image_path: Path, image_processor, model, device: str, torch_dtype: torch.dtype, process_images):
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(dtype=torch_dtype, device=device) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(dtype=torch_dtype, device=device)
    return image, image_tensor


def load_vrcd_model(model_path: str, device: str, torch_dtype: torch.dtype, vision_tower: str):
    from transformers import AutoTokenizer

    from llava.model.language_model.llava_llada import LlavaLladaForMaskedDiffusion

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = LlavaLladaForMaskedDiffusion.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map=device,
        torch_dtype=torch_dtype,
        vision_kwargs=build_vision_kwargs(vision_tower),
    )
    vision_tower_model = model.get_vision_tower()
    if not vision_tower_model.is_loaded:
        vision_tower_model.load_model(device_map=device)
    if device != "auto":
        vision_tower_model.to(device=device, dtype=torch_dtype)
    image_processor = vision_tower_model.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048
    return tokenizer, model, image_processor, context_len


def decode_text(tokenizer, output_ids: torch.Tensor) -> str:
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].lstrip("!").strip()


def resolve_default_model() -> str:
    return DEFAULT_MODEL


def validate_device(device: str) -> None:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Activate the vrcd environment before inference: conda activate vrcd"
        )


def iter_demo_items(args) -> Iterable[dict[str, Any]]:
    if args.img_path:
        image_path = Path(args.img_path)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        bundled_demo = find_demo_by_image(image_path)
        if args.question:
            question = args.question
            answer = bundled_demo.get("answer") if bundled_demo else None
            answer_text = bundled_demo.get("answer_text") if bundled_demo else None
            item_id = bundled_demo.get("id") if bundled_demo else "custom"
        elif bundled_demo is not None:
            question = format_demo_question(bundled_demo)
            answer = bundled_demo.get("answer")
            answer_text = bundled_demo.get("answer_text")
            item_id = bundled_demo.get("id")
        else:
            raise ValueError("--question is required for image paths that are not bundled examples.")
        yield {
            "id": item_id,
            "image": str(image_path),
            "question": question,
            "answer": answer,
            "answer_text": answer_text,
        }
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="LaViDa-VRCD prediction with the LaViDa reasoning checkpoint.", add_help=False)
    parser.add_argument("--model", default=resolve_default_model(), help=f"LaViDa reasoning checkpoint path or Hugging Face id. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--img-path", "--image", dest="img_path", default=str(DEFAULT_IMG_PATH.relative_to(ROOT)), help="Input image path. Bundled image paths can omit --question.")
    parser.add_argument("--question", default=None, help="Optional custom question for --img-path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--len", type=int, default=192, help="Maximum generated answer length.")
    parser.add_argument("--block-length", type=int, default=None, help="Defaults to --len.")
    parser.add_argument("--step-ratio", "--sr", dest="step_ratio", type=float, default=0.25)
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.5, help="VRCD visual-redundancy multiplier strength.")
    parser.add_argument("--window-lambda", dest="window_lambda", type=float, default=2.0, help="VRCD candidate window multiplier.")
    parser.add_argument("--vision-tower", default=os.getenv("LLADA_VISION_ENCODER", "google/siglip-so400m-patch14-384"))
    args = parser.parse_args()
    validate_device(args.device)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    _, torch_dtype = parse_dtype(args.dtype)
    tokenizer, model, image_processor, _ = load_vrcd_model(
        args.model,
        args.device,
        torch_dtype,
        args.vision_tower,
    )
    model.eval()
    model.tie_weights()
    disable_activation_checkpointing(model)
    model.to(device=args.device, dtype=torch_dtype)

    mask_token_id = MASK_TOKEN_ID
    vocab = tokenizer.get_vocab()
    if "<|mdm_mask|>" in vocab:
        mask_token_id = int(tokenizer.convert_tokens_to_ids("<|mdm_mask|>"))

    block_length = int(args.block_length or args.len)
    if int(args.len) % block_length != 0:
        raise ValueError("--len must be divisible by --block-length.")

    for item in iter_demo_items(args):
        image_path = Path(item["image"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image, image_tensor = load_image_tensor(image_path, image_processor, model, args.device, torch_dtype, process_images)
        prompt = build_prompt(str(item["question"]), "llada", DEFAULT_IMAGE_TOKEN, conv_templates)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)

        start = time.time()
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            temperature=float(args.temperature),
            max_new_tokens=int(args.len),
            block_length=block_length,
            step_ratio=float(args.step_ratio),
            tokenizer=tokenizer,
            prefix_lm=True,
            schedule="shift",
            schedule_kwargs={"shift": float(args.shift)},
            method=DEFAULT_DECODING_METHOD,
            vrcd_alpha=float(args.alpha),
            vrcd_window_lambda=float(args.window_lambda),
            mask_id=mask_token_id,
        )
        elapsed = time.time() - start
        prediction = decode_text(tokenizer, output_ids)

        print("=" * 80)
        print(f"Input: {item['id']}  method={DEFAULT_DECODING_METHOD}  len={args.len}  shift={args.shift}  step_ratio={args.step_ratio}  temperature={args.temperature}")
        print(f"VRCD: alpha={args.alpha}  window_lambda={args.window_lambda}")
        print(f"Image: {image_path}")
        if item.get("answer") is not None:
            print(f"Answer: {item['answer']} ({item.get('answer_text')})")
        print(f"Elapsed: {elapsed:.2f}s")
        print("Prediction:")
        print(prediction)


if __name__ == "__main__":
    main()
