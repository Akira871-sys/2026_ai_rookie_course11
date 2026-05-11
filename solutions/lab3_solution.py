"""
Lab 3 Solution: LoRA SFT 訓練
==============================
此為完整解答，供助教參考或學員對照。
"""

import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./qwen25vl-cord-lora"

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"


def format_to_messages(example):
    gt_parse = json.loads(example["ground_truth"])["gt_parse"]
    target = json.dumps(gt_parse, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]},
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return {"messages": messages, "images": [example["image"]]}


ds = load_dataset("naver-clova-ix/cord-v2")
train_ds = ds["train"].map(format_to_messages, remove_columns=ds["train"].column_names)
val_ds = ds["validation"].map(format_to_messages, remove_columns=ds["validation"].column_names)

IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
VIDEO_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")


def collate_fn(examples):
    texts = [
        processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
        for ex in examples
    ]
    image_inputs = [img for ex in examples for img in ex["images"]]
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token_id in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]:
        labels[labels == token_id] = -100
    batch["labels"] = labels
    return batch


# ============================================================
# TODO #1 Solution — LoRA Config
# ============================================================

peft_config = LoraConfig(
    # ✅ SOLUTION
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

# ============================================================
# TODO #2 Solution — SFTConfig
# ============================================================

args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # ✅ SOLUTION: VLM 特殊設定
    max_length=None,
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},

    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # ✅ SOLUTION: GPU 優化
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",

    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="tensorboard",
    seed=42,
)

# ============================================================
# TODO #3 Solution — VRAM 估算
# ============================================================


def estimate_vram(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_size_gb = total_params * 2 / 1e9
    lora_size_gb = trainable_params * 2 / 1e9
    optim_size_gb = trainable_params * 2 / 1e9
    grad_size_gb = trainable_params * 2 / 1e9
    activation_gb = 4.0

    total_gb = model_size_gb + lora_size_gb + optim_size_gb + grad_size_gb + activation_gb

    print(f"模型: {model_size_gb:.2f} GB | LoRA: {lora_size_gb:.4f} GB")
    print(f"Optim: {optim_size_gb:.4f} GB | Grad: {grad_size_gb:.4f} GB | Act: {activation_gb} GB")
    print(f"Total: {total_gb:.2f} GB")
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.3f}%)")
    return total_gb


# ============================================================
# 訓練
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
)

estimate_vram(trainer.model)
train_result = trainer.train()

# ============================================================
# TODO #4 Solution — Sanity Check
# ============================================================


def quick_sanity_check(trainer, val_ds, processor, n=3):
    model = trainer.model
    model.eval()

    for i in range(n):
        example = val_ds[i]

        # ✅ SOLUTION: 只取 system + user messages（不含 assistant）
        infer_messages = example["messages"][:2]

        text = processor.apply_chat_template(
            infer_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=example["images"], return_tensors="pt", padding=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
        prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        gt_text = example["messages"][2]["content"][0]["text"]
        print(f"\n--- Val {i} ---")
        print(f"GT:   {gt_text[:150]}")
        print(f"Pred: {prediction[:150]}")


quick_sanity_check(trainer, val_ds, processor)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
