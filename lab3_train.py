"""
Lab 3: LoRA SFT 訓練主流程
===========================
學習目標:
  - 配置 LoRA 微調參數
  - 設定 VLM-specific 的 SFTConfig
  - 在 16GB GPU 上完成完整訓練
  - 監控 VRAM 使用量
  - 使用 W&B 即時追蹤訓練狀態（loss、lr 曲線）

前置要求:
  - 已完成 wandb login（見 README.md）

執行方式:
    uv run python lab3_train.py

預期:
  - VRAM peak: 11-12 GB
  - Loss: 從 ~2.5 降到 ~0.4
  - Adapter 大小: ~30 MB
"""

import json

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

# ============================================================
# Section 1: 初始化 Weights & Biases
# ============================================================

wandb.init(
    project="vlm-sft-lab",
    name="qwen25vl-cord-lora",
    config={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "task": "receipt-extraction",
        "dataset": "naver-clova-ix/cord-v2",
        "method": "LoRA SFT",
    },
)
print("W&B 初始化完成！可至 https://wandb.ai 查看即時訓練狀態。")

# ============================================================
# Section 2: 載入模型 + Processor
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./qwen25vl-cord-lora"

print("載入模型與 processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print(f"模型載入完成: {MODEL_ID}")

# ============================================================
# Section 3: 載入並格式化資料集（沿用 Lab 2 的邏輯）
# ============================================================

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"


def format_to_messages(example):
    gt_parse = json.loads(example["ground_truth"])["gt_parse"]
    target = json.dumps(gt_parse, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": target}]},
    ]
    return {"messages": messages, "images": [example["image"]]}


print("載入與格式化資料集...")
ds = load_dataset("naver-clova-ix/cord-v2")
train_ds = ds["train"].map(format_to_messages, remove_columns=ds["train"].column_names)
val_ds = ds["validation"].map(format_to_messages, remove_columns=ds["validation"].column_names)
print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

# ============================================================
# Section 4: Collate Function（Lab 2 的完成版）
# ============================================================

IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
VIDEO_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")


def collate_fn(examples):
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
    for token_id in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]:
        labels[labels == token_id] = -100

    batch["labels"] = labels
    return batch


# ============================================================
# Section 5: TODO #1 — LoRA Config 🟡
# ============================================================

peft_config = LoraConfig(
    # TODO 1.1: 設定 rank 🟢
    # Hint: r=8~16 對 3B 模型已足夠，越大 = 越多參數、效果更好但更耗 VRAM
    r=___,

    # TODO 1.2: 設定 alpha 🟢
    # Hint: 通常設 2*r，控制 LoRA 的 scaling factor (alpha/r)
    lora_alpha=___,

    # TODO 1.3: 設定 dropout 🟢
    # Hint: 0.05 是常見值，防止小資料集過擬合
    lora_dropout=___,

    # TODO 1.4: bias 設定 🟢
    # Hint: "none" — 不訓練 bias，因為效果有限但增加複雜度
    bias="___",

    # TODO 1.5: 指定要 LoRA 的層 🟡
    # Hint: 只調 LLM decoder 的 attention layers
    # Qwen2.5-VL 的 attention 層名稱: q_proj, k_proj, v_proj, o_proj
    # 不要調 vision encoder (名字裡有 "visual")
    target_modules=[___, ___, ___, ___],

    task_type="CAUSAL_LM",
)

print(f"\nLoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")

# ============================================================
# Section 6: TODO #2 — SFTConfig 🟡
# ============================================================

args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # === VLM 特殊設定（這是與純文字 SFT 的關鍵差異！）===

    # TODO 2.1: VLM 不能用固定 max_length 截斷（會破壞 image tokens）🟢
    # Hint: 設成 None，讓每個 batch 自然決定長度
    max_length=___,

    # TODO 2.2: 保留 images 欄位 🟢
    # Hint: 預設 trainer 會移除非標準欄位，但我們的 collator 需要 images
    remove_unused_columns=___,

    # TODO 2.3: 告訴 trainer 跳過內建的 dataset 前處理 🟢
    # Hint: 因為我們用自定義 collator 處理一切
    dataset_kwargs={"skip_prepare_dataset": ___},

    # === 一般 SFT 訓練參數 ===
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 有效 batch = 1 * 8 = 8
    num_train_epochs=2,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",

    # === 16GB GPU 必備優化 ===

    # TODO 2.4: 用 bfloat16 混合精度訓練 🟢
    # Hint: Ada / Ampere 以上架構都支援 bf16
    bf16=___,

    # TODO 2.5: 開 gradient checkpointing 省 VRAM 🟢
    # Hint: 用空間換時間，可省 30-50% 激活值 VRAM
    gradient_checkpointing=___,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # TODO 2.6: 8-bit optimizer 省 optimizer state VRAM 🟢
    # Hint: "adamw_8bit" 或 "paged_adamw_8bit"
    optim="___",

    # === 監控與儲存 ===
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="wandb",
    seed=42,
)

# ============================================================
# Section 7: TODO #3 — VRAM 估算 🟡
# ============================================================


def estimate_vram(model):
    """估算訓練期間 VRAM 用量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # TODO 3.1: 計算各部分佔用 🟡
    # bf16 = 2 bytes/param
    model_size_gb = total_params * 2 / 1e9
    lora_size_gb = trainable_params * 2 / 1e9

    # 8-bit AdamW: 每個可訓練參數約 2 bytes state
    optim_size_gb = trainable_params * 2 / 1e9

    # 梯度: bf16 = 2 bytes/trainable param
    grad_size_gb = trainable_params * 2 / 1e9

    # 激活值: 經驗值，取決於 batch_size + seq_length + image_tokens
    activation_gb = 4.0

    total_gb = model_size_gb + lora_size_gb + optim_size_gb + grad_size_gb + activation_gb

    print("\n" + "=" * 60)
    print("VRAM 使用估算")
    print("=" * 60)
    print(f"  模型權重 (bf16):   {model_size_gb:.2f} GB")
    print(f"  LoRA adapter:      {lora_size_gb:.4f} GB")
    print(f"  Optimizer state:   {optim_size_gb:.4f} GB")
    print(f"  梯度:              {grad_size_gb:.4f} GB")
    print(f"  激活值 (估算):     {activation_gb:.2f} GB")
    print(f"  ─────────────────────────────")
    print(f"  預估總用量:        {total_gb:.2f} GB")
    print(f"\n  可訓練參數: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.3f}%)")

    if total_gb > 15:
        print("\n  ⚠️  可能超過 16GB！建議:")
        print("     - 降低 batch_size 到 1")
        print("     - 降低 max_pixels")
        print("     - 或啟用 QLoRA (4-bit)")
    else:
        print(f"\n  ✅ 預計可在 16GB GPU 上運行（剩餘 ~{16 - total_gb:.1f} GB buffer）")

    return total_gb


# 在加 LoRA 之前先看 base model
print("\n--- Base model 參數 ---")
estimate_vram(model)

# ============================================================
# Section 8: 建立 SFTTrainer
# ============================================================

print("\n建立 SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor,
)

# LoRA 加上後再估算一次
print("\n--- 加上 LoRA 後 ---")
estimate_vram(trainer.model)

# ============================================================
# Section 9: 啟動訓練
# ============================================================

wandb.config.update({
    "lora_r": peft_config.r,
    "lora_alpha": peft_config.lora_alpha,
    "epochs": args.num_train_epochs,
    "batch_size": args.per_device_train_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "optimizer": args.optim,
})

print("\n" + "=" * 60)
print("開始訓練！")
print("=" * 60)
print(f"  Output dir:   {OUTPUT_DIR}")
print(f"  Epochs:       {args.num_train_epochs}")
print(f"  Batch size:   {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
print(f"  Learning rate: {args.learning_rate}")
print(f"  LoRA rank:    {peft_config.r}")
print()
print("提示: 訓練過程即時同步到 W&B，開啟瀏覽器查看:")
print(f"  {wandb.run.get_url()}")
print()
print("提示: 用 nvidia-smi -l 1 監控 GPU 使用量")
print("=" * 60)

train_result = trainer.train()

# ============================================================
# Section 10: 訓練結果
# ============================================================

print("\n" + "=" * 60)
print("訓練完成！")
print("=" * 60)
metrics = train_result.metrics
print(f"  Train loss:     {metrics.get('train_loss', 'N/A')}")
print(f"  Train runtime:  {metrics.get('train_runtime', 0):.0f} seconds")
print(f"  Train samples/s: {metrics.get('train_samples_per_second', 0):.2f}")

# ============================================================
# Section 11: TODO #4 — Quick Sanity Check 🟡
# ============================================================


def quick_sanity_check(trainer, val_ds, processor, n=3):
    """
    訓練後快速確認模型是否真的學到東西。
    對 validation set 的幾筆資料做推論，並對照 ground truth。
    """
    model = trainer.model
    model.eval()

    print("\n" + "=" * 60)
    print(f"Sanity Check（{n} 筆 validation data）")
    print("=" * 60)

    for i in range(n):
        example = val_ds[i]

        # TODO 4.1: 建立推論用的 messages（只有 system + user，不含 assistant）🟡
        # Hint: 複製 example["messages"] 的前兩個 turn
        infer_messages = ___

        # TODO 4.2: 套 chat template + processor 處理 🟡
        text = processor.apply_chat_template(
            infer_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=example["images"],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        # TODO 4.3: generate 並 decode 🟢
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

        generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
        prediction = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Ground truth
        gt_text = example["messages"][2]["content"][0]["text"]

        print(f"\n--- Val sample {i} ---")
        print(f"GT (前 150 字):   {gt_text[:150]}")
        print(f"Pred (前 150 字): {prediction[:150]}")
        print()


quick_sanity_check(trainer, val_ds, processor)

# ============================================================
# Section 12: 保存 Adapter
# ============================================================

print("\n保存 LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

import os
adapter_size = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith((".safetensors", ".bin"))
)
print(f"  Adapter 已存到: {OUTPUT_DIR}")
print(f"  Adapter 大小:   {adapter_size / 1e6:.1f} MB")

# ============================================================
# Section 13: 上傳結果到 W&B 並結束
# ============================================================

wandb.log({
    "train/final_loss": metrics.get("train_loss", 0),
    "train/runtime_sec": metrics.get("train_runtime", 0),
    "train/samples_per_sec": metrics.get("train_samples_per_second", 0),
    "adapter_size_mb": adapter_size / 1e6,
})

wandb.finish()
print("\nW&B run 已結束。完整訓練記錄可至 https://wandb.ai 查看。")
