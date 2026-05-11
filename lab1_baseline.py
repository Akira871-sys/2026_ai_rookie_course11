"""
Lab 1: 資料探索 + Zero-shot Baseline
=====================================
學習目標:
  - 認識 CORD-v2 收據資料集
  - 體驗 Qwen2.5-VL-3B 原始 zero-shot 能力
  - 建立量化 baseline（Field F1）

執行方式:
    uv run python lab1_baseline.py
"""

import json
import random

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm.auto import tqdm

# ============================================================
# Section 1: 設定
# ============================================================

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
NUM_EVAL_SAMPLES = 100

random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# Section 2: 載入 CORD-v2 資料集
# ============================================================

print("載入 CORD-v2 資料集...")
ds = load_dataset("naver-clova-ix/cord-v2")
print(f"  Train:      {len(ds['train'])} 筆")
print(f"  Validation: {len(ds['validation'])} 筆")
print(f"  Test:       {len(ds['test'])} 筆")

# ============================================================
# Section 3: TODO #1 — 視覺化資料 🟢
# ============================================================
# 從 train set 隨機取 5 筆，顯示圖片 + 解析後的 ground truth
#
# Hint 1: example["image"] 就是 PIL Image
# Hint 2: example["ground_truth"] 是 JSON string，需要 json.loads()
# Hint 3: 真正的 GT 在 parsed result 的 "gt_parse" key 底下


def visualize_samples(dataset, n=5):
    """隨機視覺化 n 筆資料"""
    samples = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(1, n, figsize=(20, 6))

    for i, idx in enumerate(samples):
        example = dataset[idx]

        # === START TODO #1 ===
        image = ___  # 取出圖片
        gt = json.loads(___)[___]  # 取出並解析 ground truth
        # === END TODO #1 ===

        axes[i].imshow(image)
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis("off")

        print(f"\n=== Sample {idx} ===")
        print(json.dumps(gt, ensure_ascii=False, indent=2)[:500])

    plt.tight_layout()
    plt.savefig("lab1_samples.png", dpi=100)
    plt.show()
    print("\n圖片已存為 lab1_samples.png")


visualize_samples(ds["train"])

# ============================================================
# Section 4: 載入 Qwen2.5-VL-3B 模型
# ============================================================

print(f"\n載入模型: {MODEL_ID}...")
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
model.eval()
print("模型載入完成。")

# ============================================================
# Section 5: TODO #2 — Zero-shot 推論函式 🟡
# ============================================================


def predict_receipt(image, model, processor, max_new_tokens=512):
    """
    對單張收據圖片做 zero-shot JSON 抽取。

    Args:
        image: PIL Image
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        max_new_tokens: 最大生成 token 數

    Returns:
        str: 模型生成的文字
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                # TODO 2.1: 寫一個明確的 prompt，要求模型以 JSON 格式抽取收據資訊 🟢
                # 提示: 告訴模型你要什麼格式、哪些欄位
                {"type": "text", "text": "___"},
            ],
        }
    ]

    # 套用 chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # TODO 2.2: 用 process_vision_info 從 messages 取出影像輸入 🟢
    # Hint: process_vision_info 回傳 (image_inputs, video_inputs)
    image_inputs, _ = ___(messages)

    # TODO 2.3: 用 processor 把 text + images 包成 tensor batch 🟡
    # Hint: processor(text=[...], images=[...], ...)
    inputs = processor(
        text=___,
        images=___,
        return_tensors="pt",
        padding=True,
    )

    inputs = inputs.to(model.device)

    # 推論
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # TODO 2.4: 截掉 input 部分，只 decode 新生成的 tokens 🟢
    # Hint: output_ids 包含 input + generated，需要用 inputs["input_ids"] 的長度來切
    generated_ids = [
        output_ids[i][len(inputs["input_ids"][i]):]
        for i in range(len(output_ids))
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return output_text


# ============================================================
# Section 6: 跑幾筆 zero-shot 預測看看
# ============================================================

print("\n" + "=" * 60)
print("Zero-shot 推論範例（3 筆）")
print("=" * 60)

for i in range(3):
    example = ds["test"][i]
    image = example["image"]
    gt = json.loads(example["ground_truth"])["gt_parse"]

    pred_text = predict_receipt(image, model, processor)

    print(f"\n--- Test sample {i} ---")
    print(f"Ground Truth (前 200 字):\n{json.dumps(gt, ensure_ascii=False)[:200]}")
    print(f"\nPrediction (前 200 字):\n{pred_text[:200]}")

# ============================================================
# Section 7: TODO #3 — Field F1 評估函式 🟡
# ============================================================


def flatten_dict(d, prefix=""):
    """
    把 nested dict/list 攤平成 {path: value} 格式。

    Examples:
        {"menu": [{"nm": "Coffee"}]}
        → {"menu.0.nm": "coffee"}
    """
    items = {}
    # TODO 3.1: 遞迴展開 dict 與 list 🟡
    # Hint: isinstance(d, dict) → 遍歷 k, v
    #        isinstance(d, list) → 遍歷 i, v
    #        否則 → items[prefix] = str(d).strip().lower()
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(___)  # 遞迴呼叫 flatten_dict
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            items.update(___)  # 遞迴呼叫 flatten_dict
    else:
        items[prefix] = str(d).strip().lower()
    return items


def field_f1(pred_dict, gt_dict):
    """
    計算 field-level F1。

    把 nested dict flatten 後，比較 key-value pairs。
    正確 = 同 key 且同 value。

    Returns:
        dict: {"precision": float, "recall": float, "f1": float}
    """
    pred_flat = flatten_dict(pred_dict)
    gt_flat = flatten_dict(gt_dict)

    # TODO 3.2: 計算 TP, FP, FN 🟢
    # TP = pred 和 gt 都有的 key，且 value 相同
    # FP = pred 有但 gt 沒有，或 value 不同
    # FN = gt 有但 pred 沒有
    tp = ___
    fp = ___
    fn = ___

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================
# Section 8: 跑全 test set 評估 zero-shot baseline
# ============================================================


def parse_json_safe(text):
    """嘗試從模型輸出中解析 JSON"""
    text = text.strip()
    # 嘗試找 JSON block
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate_zero_shot(model, processor, dataset, n_samples=NUM_EVAL_SAMPLES):
    """對 test set 跑 zero-shot 評估"""
    print(f"\n評估 zero-shot（{n_samples} 筆）...")

    results = []
    parse_success = 0

    for i in tqdm(range(min(n_samples, len(dataset)))):
        example = dataset[i]
        image = example["image"]
        gt = json.loads(example["ground_truth"])["gt_parse"]

        pred_text = predict_receipt(image, model, processor)
        pred_dict = parse_json_safe(pred_text)

        if pred_dict is not None:
            parse_success += 1
            score = field_f1(pred_dict, gt)
        else:
            score = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        results.append(score)

    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print("Zero-shot Baseline 結果")
    print("=" * 60)
    print(f"  JSON parse 成功率: {parse_success}/{n_samples} ({parse_success/n_samples*100:.1f}%)")
    print(f"  Avg Precision:     {avg_precision:.4f}")
    print(f"  Avg Recall:        {avg_recall:.4f}")
    print(f"  Avg Field F1:      {avg_f1:.4f}")
    print("=" * 60)

    return {
        "parse_rate": parse_success / n_samples,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "details": results,
    }


baseline_results = evaluate_zero_shot(model, processor, ds["test"])

# 存下 baseline 結果供 Lab 4 對照
with open("baseline_results.json", "w") as f:
    json.dump(
        {k: v for k, v in baseline_results.items() if k != "details"},
        f,
        indent=2,
    )
print("\nBaseline 結果已存為 baseline_results.json")

# ============================================================
# Section 9: 反思題
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  反思題（請在下方寫下你的觀察）                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Zero-shot 的 Field F1 大約是多少？JSON parse 成功率？     ║
║                                                              ║
║  2. 觀察失敗案例，模型主要錯在哪？                           ║
║     (格式錯？欄位漏？幻覺？)                                 ║
║                                                              ║
║  3. 為什麼通用 VLM 在這個任務上表現不好？                    ║
║     SFT 需要解決什麼問題？                                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

你的回答:
---
(在此處寫下你的觀察與反思)
---
""")
