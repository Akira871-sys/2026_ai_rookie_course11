"""
Lab 2: 資料前處理 + 自定義 Collator
====================================
學習目標:
  - 將 CORD-v2 raw dataset 轉成 TRL SFTTrainer 接受的 messages 格式
  - 實作 VLM SFT 必備的客製 collate function
  - 正確處理 image token masking（關鍵！）
  - 估算合適的 max_pixels 設定

執行方式:
    uv run python lab2_data_collator.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor

# ============================================================
# Section 1: 載入資料集與 processor
# ============================================================

print("載入 CORD-v2 資料集...")
ds = load_dataset("naver-clova-ix/cord-v2")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)

# ============================================================
# Section 2: Messages 模板設計（講解）
# ============================================================
#
# TRL SFTTrainer 要求資料格式為:
# {
#     "messages": [
#         {"role": "system", "content": [...]},
#         {"role": "user",   "content": [{"type": "image"}, {"type": "text", ...}]},
#         {"role": "assistant", "content": [{"type": "text", ...}]},
#     ],
#     "images": [PIL.Image, ...]
# }
#
# 重點:
# 1. system prompt 設定角色與輸出格式
# 2. user message 包含圖片占位符 + 指令
# 3. assistant message 包含 ground truth JSON（訓練 target）

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"

INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"

# ============================================================
# Section 3: TODO #1 — format_to_messages 函式 🟡
# ============================================================


def format_to_messages(example):
    """
    把 CORD-v2 格式轉成 TRL SFTTrainer 接受的格式。

    輸入 (CORD-v2 格式):
        {
            "image": PIL.Image,
            "ground_truth": '{"gt_parse": {...}}'  # JSON string
        }

    輸出 (TRL 格式):
        {
            "messages": [system_msg, user_msg, assistant_msg],
            "images": [PIL.Image]
        }
    """
    # TODO 1.1: parse ground truth 並轉成漂亮的 JSON 字串 🟢
    # Hint: ground_truth 是字串，先 json.loads() 再取 "gt_parse" key
    gt_parse = json.loads(___)[___]
    target = json.dumps(___, ensure_ascii=False, indent=2)

    # TODO 1.2: 建立 3-turn messages (system / user / assistant) 🟡
    # Hint:
    #   - system: 告訴模型它的角色（用上面的 SYSTEM_PROMPT）
    #   - user: 包含一個 {"type": "image"} 和一個 {"type": "text", "text": INSTRUCTION}
    #   - assistant: 包含 target JSON 字串
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ___}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": ___},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": ___}],
        },
    ]

    return {
        "messages": messages,
        "images": [example["image"]],
    }


# ============================================================
# Section 4: 驗證格式轉換
# ============================================================

print("\n驗證 format_to_messages:")
sample = ds["train"][0]
formatted = format_to_messages(sample)

print(f"  messages 數量: {len(formatted['messages'])}")
print(f"  images 數量:   {len(formatted['images'])}")
print(f"  system role:   {formatted['messages'][0]['role']}")
print(f"  user role:     {formatted['messages'][1]['role']}")
print(f"  assistant role: {formatted['messages'][2]['role']}")
print(f"  assistant 回覆前 100 字: {formatted['messages'][2]['content'][0]['text'][:100]}...")

# ============================================================
# Section 5: 套用 .map() 處理全資料集
# ============================================================

print("\n處理全資料集...")
train_ds = ds["train"].map(
    format_to_messages,
    remove_columns=ds["train"].column_names,
)
val_ds = ds["validation"].map(
    format_to_messages,
    remove_columns=ds["validation"].column_names,
)
print(f"  Train: {len(train_ds)} 筆")
print(f"  Val:   {len(val_ds)} 筆")

# ============================================================
# Section 6: TODO #2 — Collate Function 主體 🟡
# ============================================================


def collate_fn(examples):
    """
    把多個 examples 組成一個 batch。

    重點:
    1. 文字部分用 processor.apply_chat_template 套上對話模板
    2. 影像部分用 processor 一起處理
    3. labels 需要 mask 掉不該算 loss 的部分（padding）

    Args:
        examples: list of {"messages": [...], "images": [...]}

    Returns:
        dict with input_ids, attention_mask, pixel_values, labels, ...
    """
    # TODO 2.1: 對每個 example 套 chat template，得到文字字串 🟢
    # Hint: apply_chat_template 的參數:
    #   - tokenize=False (我們後面自己 tokenize)
    #   - add_generation_prompt=False (訓練時不要加)
    texts = [
        processor.apply_chat_template(
            ex[___],           # 取 messages
            tokenize=___,      # False
            add_generation_prompt=___,  # False
        )
        for ex in examples
    ]

    # TODO 2.2: 收集所有圖片 🟢
    # Hint: 每個 example 的 "images" 是 list，要 flatten 成一個大 list
    image_inputs = [img for ex in examples for img in ex[___]]

    # TODO 2.3: 用 processor 一次處理 text + image 🟢
    # Hint: processor(text=..., images=..., return_tensors="pt", padding=True)
    batch = processor(
        text=___,
        images=___,
        return_tensors="pt",
        padding=True,
    )

    # TODO 2.4: 準備 labels 🟢
    # 從 input_ids 複製，但 padding token 設為 -100（不算 loss）
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    batch["labels"] = labels
    return batch


# ============================================================
# Section 7: 測試 collate_fn
# ============================================================

print("\n測試 collate_fn...")
test_batch = collate_fn([train_ds[0], train_ds[1]])
print(f"  input_ids shape:  {test_batch['input_ids'].shape}")
print(f"  labels shape:     {test_batch['labels'].shape}")
if "pixel_values" in test_batch:
    print(f"  pixel_values shape: {test_batch['pixel_values'].shape}")
print(f"  -100 比例 (labels): {(test_batch['labels'] == -100).float().mean():.2%}")

# ============================================================
# Section 8: TODO #3 — Image Token Masking（關鍵坑！）🔴
# ============================================================
#
# 為什麼需要 mask image tokens?
# Qwen2.5-VL 會把圖片展開成大量的 <|image_pad|> token。
# 這些 token 是「占位符」，不攜帶語義資訊，
# 如果算進 loss，會讓模型學到錯誤的東西且拖慢收斂。

# 先取得 image-related token ids
IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
VIDEO_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")


def collate_fn_v2(examples):
    """
    改進版 collate_fn:
    1. 同 v1 的處理
    2. 額外 mask 掉 image/video pad tokens
    3. (Bonus) 只在 assistant response 部分算 loss
    """
    # --- 同 v1 部分 ---
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

    # TODO 3.1: mask 掉 image/video pad tokens 🟢
    # Hint: 把 labels 中等於 IMAGE_TOKEN_ID 或 VIDEO_TOKEN_ID 的位置設為 -100
    for token_id in [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]:
        labels[labels == ___] = -100

    # TODO 3.2 (Bonus ⚪): 只在 assistant response 部分算 loss
    # 把 system + user prompt 的 token 也設為 -100
    # 提示: 找出 "<|im_start|>assistant\n" 在 input_ids 中的位置
    #        該位置之前的 labels 都設為 -100
    #
    # assistant_token_pattern = processor.tokenizer.encode(
    #     "<|im_start|>assistant\n", add_special_tokens=False
    # )
    # for i in range(labels.shape[0]):
    #     # 找到 pattern 的起始位置
    #     input_ids_list = batch["input_ids"][i].tolist()
    #     ...找 pattern 位置...
    #     labels[i, :start_pos] = -100

    batch["labels"] = labels
    return batch


# ============================================================
# Section 9: 驗證 image token masking
# ============================================================

print("\n驗證 image token masking (collate_fn_v2):")
test_batch_v2 = collate_fn_v2([train_ds[0], train_ds[1]])
labels_v2 = test_batch_v2["labels"]

total_tokens = labels_v2.numel()
masked_tokens = (labels_v2 == -100).sum().item()
active_tokens = total_tokens - masked_tokens

print(f"  總 token 數:      {total_tokens}")
print(f"  被 mask 的 token: {masked_tokens} ({masked_tokens/total_tokens*100:.1f}%)")
print(f"  參與 loss 的:     {active_tokens} ({active_tokens/total_tokens*100:.1f}%)")

# 視覺化 labels mask（第一筆）
print("\n  Labels 視覺化（前 100 tokens）:")
first_labels = labels_v2[0][:100].tolist()
mask_viz = ["█" if t == -100 else "░" for t in first_labels]
print(f"  {''.join(mask_viz)}")
print(f"  █ = masked (-100)    ░ = active (算 loss)")

# ============================================================
# Section 10: TODO #4 — max_pixels 估算 🟡
# ============================================================

print("\n" + "=" * 60)
print("max_pixels 估算")
print("=" * 60)

# TODO 4.1: 計算 train set 每張圖片的 pixel 數 🟢
# Hint: image.width * image.height
pixels = [___ for ex in ds["train"]]

# TODO 4.2: 顯示分位數 🟢
print(f"  min: {np.min(pixels)/1e6:.2f} MP")
print(f"  p25: {np.percentile(pixels, 25)/1e6:.2f} MP")
print(f"  p50: {np.percentile(pixels, 50)/1e6:.2f} MP")
print(f"  p75: {np.percentile(pixels, 75)/1e6:.2f} MP")
print(f"  p95: {np.percentile(pixels, 95)/1e6:.2f} MP")
print(f"  max: {np.max(pixels)/1e6:.2f} MP")

# TODO 4.3: 根據 16GB GPU + batch_size=2 決定 max_pixels 🟡
# 公式推導:
#   - Qwen2.5-VL 每 28×28 patch 算 1 個 visual token
#   - 經驗: 3B + LoRA，單張圖 visual tokens 不超過 1024 比較穩
#   - max_pixels = num_visual_tokens * 28 * 28
#
# 根據你的 p95 統計和 GPU 限制，選擇合適的 num_visual_tokens
num_visual_tokens = ___  # 建議 768~1024
max_pixels_chosen = num_visual_tokens * 28 * 28

print(f"\n  建議 num_visual_tokens = {num_visual_tokens}")
print(f"  建議 max_pixels = {max_pixels_chosen:,} ({max_pixels_chosen/1e6:.2f} MP)")
print(f"  這大約是 {int(np.sqrt(max_pixels_chosen))}×{int(np.sqrt(max_pixels_chosen))} 的正方形圖片")

# 視覺化 pixel 分佈
plt.figure(figsize=(10, 4))
plt.hist([p / 1e6 for p in pixels], bins=50, edgecolor="black", alpha=0.7)
plt.axvline(max_pixels_chosen / 1e6, color="red", linestyle="--", label=f"max_pixels = {max_pixels_chosen/1e6:.2f} MP")
plt.xlabel("Megapixels")
plt.ylabel("Count")
plt.title("CORD-v2 Train Set — Image Size Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("lab2_pixel_distribution.png", dpi=100)
plt.show()
print("\n圖表已存為 lab2_pixel_distribution.png")
