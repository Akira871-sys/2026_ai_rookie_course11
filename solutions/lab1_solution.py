"""
Lab 1 Solution: 資料探索 + Zero-shot Baseline
==============================================
此為完整解答，供助教參考或學員對照。
"""

import json
import random

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm.auto import tqdm

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
NUM_EVAL_SAMPLES = 100

random.seed(SEED)
torch.manual_seed(SEED)

ds = load_dataset("naver-clova-ix/cord-v2")

# ============================================================
# TODO #1 Solution — 視覺化資料
# ============================================================


def visualize_samples(dataset, n=5):
    samples = random.sample(range(len(dataset)), n)
    fig, axes = plt.subplots(1, n, figsize=(20, 6))

    for i, idx in enumerate(samples):
        example = dataset[idx]

        # ✅ SOLUTION
        image = example["image"]
        gt = json.loads(example["ground_truth"])["gt_parse"]

        axes[i].imshow(image)
        axes[i].set_title(f"Sample {idx}")
        axes[i].axis("off")

        print(f"\n=== Sample {idx} ===")
        print(json.dumps(gt, ensure_ascii=False, indent=2)[:500])

    plt.tight_layout()
    plt.savefig("lab1_samples.png", dpi=100)
    plt.show()


visualize_samples(ds["train"])

# ============================================================
# 載入模型
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

# ============================================================
# TODO #2 Solution — Zero-shot 推論函式
# ============================================================


def predict_receipt(image, model, processor, max_new_tokens=512):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                # ✅ SOLUTION: 明確要求 JSON 格式
                {
                    "type": "text",
                    "text": (
                        "Extract all information from this receipt image and output as a JSON object. "
                        "Include fields like: store name, address, phone, date, menu items "
                        "(name, quantity, price), subtotal, tax, total, and payment method if visible."
                    ),
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # ✅ SOLUTION: process_vision_info
    image_inputs, _ = process_vision_info(messages)

    # ✅ SOLUTION: processor 打包
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # ✅ SOLUTION: 截掉 input 部分
    generated_ids = [
        output_ids[i][len(inputs["input_ids"][i]):] for i in range(len(output_ids))
    ]

    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text


# ============================================================
# TODO #3 Solution — Field F1 評估
# ============================================================


def flatten_dict(d, prefix=""):
    items = {}
    # ✅ SOLUTION: 遞迴展開
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_dict(v, new_key))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            items.update(flatten_dict(v, new_key))
    else:
        items[prefix] = str(d).strip().lower()
    return items


def field_f1(pred_dict, gt_dict):
    pred_flat = flatten_dict(pred_dict)
    gt_flat = flatten_dict(gt_dict)

    # ✅ SOLUTION: TP / FP / FN
    tp = sum(1 for k in pred_flat if k in gt_flat and pred_flat[k] == gt_flat[k])
    fp = sum(1 for k in pred_flat if k not in gt_flat or pred_flat[k] != gt_flat[k])
    fn = sum(1 for k in gt_flat if k not in pred_flat or gt_flat[k] != pred_flat.get(k))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================
# 完整 Baseline 評估
# ============================================================


def parse_json_safe(text):
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def evaluate_zero_shot(model, processor, dataset, n_samples=NUM_EVAL_SAMPLES):
    results = []
    parse_success = 0

    for i in tqdm(range(min(n_samples, len(dataset)))):
        example = dataset[i]
        gt = json.loads(example["ground_truth"])["gt_parse"]
        pred_text = predict_receipt(example["image"], model, processor)
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

    print(f"\nJSON parse 成功率: {parse_success}/{n_samples} ({parse_success/n_samples*100:.1f}%)")
    print(f"Avg Precision: {avg_precision:.4f}")
    print(f"Avg Recall:    {avg_recall:.4f}")
    print(f"Avg Field F1:  {avg_f1:.4f}")

    return {"parse_rate": parse_success / n_samples, "precision": avg_precision, "recall": avg_recall, "f1": avg_f1}


baseline_results = evaluate_zero_shot(model, processor, ds["test"])

with open("baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)
