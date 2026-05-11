"""
Lab 2 Solution: 資料前處理 + 自定義 Collator
=============================================
此為完整解答，供助教參考或學員對照。
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor

ds = load_dataset("naver-clova-ix/cord-v2")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"

# ============================================================
# TODO #1 Solution — format_to_messages
# ============================================================


def format_to_messages(example):
    # ✅ SOLUTION
    gt_parse = json.loads(example["ground_truth"])["gt_parse"]
    target = json.dumps(gt_parse, ensure_ascii=False, indent=2)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        },
    ]

    return {
        "messages": messages,
        "images": [example["image"]],
    }


train_ds = ds["train"].map(format_to_messages, remove_columns=ds["train"].column_names)
val_ds = ds["validation"].map(format_to_messages, remove_columns=ds["validation"].column_names)

# ============================================================
# TODO #2 Solution — collate_fn
# ============================================================


def collate_fn(examples):
    # ✅ SOLUTION
    texts = [
        processor.apply_chat_template(
            ex["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        for ex in examples
    ]

    image_inputs = [img for ex in examples for img in ex["images"]]

    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch


# ============================================================
# TODO #3 Solution — Image Token Masking
# ============================================================

IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
VIDEO_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")


def collate_fn_v2(examples):
    texts = [
        processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        for ex in examples
    ]

    image_inputs = [img for ex in examples for img in ex["images"]]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # ✅ SOLUTION: mask image/video pad tokens
    for token_id in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]:
        labels[labels == token_id] = -100

    # ✅ BONUS SOLUTION: 只在 assistant 部分算 loss
    assistant_start_tokens = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    for i in range(labels.shape[0]):
        input_ids_list = batch["input_ids"][i].tolist()
        # 找 assistant 起始位置
        found = False
        for pos in range(len(input_ids_list) - len(assistant_start_tokens)):
            if input_ids_list[pos : pos + len(assistant_start_tokens)] == assistant_start_tokens:
                # mask 掉 assistant start 之前的所有 token（含 assistant start 本身）
                labels[i, : pos + len(assistant_start_tokens)] = -100
                found = True
                break
        if not found:
            # 如果找不到（不應發生），全部不算 loss
            labels[i, :] = -100

    batch["labels"] = labels
    return batch


# 驗證
test_batch = collate_fn_v2([train_ds[0], train_ds[1]])
total = test_batch["labels"].numel()
masked = (test_batch["labels"] == -100).sum().item()
print(f"Masked: {masked}/{total} ({masked/total*100:.1f}%)")

# ============================================================
# TODO #4 Solution — max_pixels 估算
# ============================================================

# ✅ SOLUTION
pixels = [ex["image"].width * ex["image"].height for ex in ds["train"]]

print(f"min: {np.min(pixels)/1e6:.2f} MP")
print(f"p50: {np.percentile(pixels, 50)/1e6:.2f} MP")
print(f"p95: {np.percentile(pixels, 95)/1e6:.2f} MP")
print(f"max: {np.max(pixels)/1e6:.2f} MP")

# ✅ SOLUTION: 選擇 1024 visual tokens
num_visual_tokens = 1024
max_pixels_chosen = num_visual_tokens * 28 * 28
print(f"\n建議 max_pixels = {max_pixels_chosen:,} ({max_pixels_chosen/1e6:.2f} MP)")
