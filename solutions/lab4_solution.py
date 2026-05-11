"""
Lab 4 Solution: 評估 + 結果分析
================================
此為完整解答，供助教參考或學員對照。
"""

import json
import os

import edit_distance
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm.auto import tqdm

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = "./qwen25vl-cord-lora"
NUM_EVAL_SAMPLES = 100

processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

ds = load_dataset("naver-clova-ix/cord-v2")
test_ds = ds["test"]

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"


def predict_single(image, model, processor, max_new_tokens=768):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": INSTRUCTION},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)


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


# ============================================================
# TODO #1 Solution — Batch 推論
# ============================================================


def batch_predict(model, processor, dataset, n_samples=NUM_EVAL_SAMPLES):
    results = []

    for i in tqdm(range(min(n_samples, len(dataset))), desc="SFT 推論"):
        example = dataset[i]
        image = example["image"]
        gt = json.loads(example["ground_truth"])["gt_parse"]

        # ✅ SOLUTION
        pred_text = predict_single(image, model, processor)
        pred_dict = parse_json_safe(pred_text)

        results.append({
            "index": i,
            "pred_text": pred_text,
            "pred_dict": pred_dict,
            "gt_dict": gt,
        })

    return results


sft_results = batch_predict(model, processor, test_ds)

# ============================================================
# Metrics
# ============================================================


def flatten_dict(d, prefix=""):
    items = {}
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
    tp = sum(1 for k in pred_flat if k in gt_flat and pred_flat[k] == gt_flat[k])
    fp = sum(1 for k in pred_flat if k not in gt_flat or pred_flat[k] != gt_flat[k])
    fn = sum(1 for k in gt_flat if k not in pred_flat or gt_flat[k] != pred_flat.get(k))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1}


# ============================================================
# TODO #2 Solution — Tree Edit Distance
# ============================================================


def normalize_tree(d):
    # ✅ SOLUTION
    if isinstance(d, dict):
        return {k: normalize_tree(v) for k, v in sorted(d.items())}
    elif isinstance(d, list):
        return [normalize_tree(item) for item in d]
    elif isinstance(d, str):
        return d.strip().lower()
    else:
        return d


def tree_edit_similarity(pred, gt):
    pred_str = json.dumps(normalize_tree(pred), sort_keys=True, ensure_ascii=False)
    gt_str = json.dumps(normalize_tree(gt), sort_keys=True, ensure_ascii=False)

    # ✅ SOLUTION
    sm = edit_distance.SequenceMatcher(a=pred_str, b=gt_str)
    distance = sm.distance()
    max_len = max(len(pred_str), len(gt_str))
    normalized_distance = distance / max_len

    return 1 - normalized_distance


# ============================================================
# 計算結果
# ============================================================

f1_scores = []
ted_scores = []
parse_success = 0

for r in sft_results:
    if r["pred_dict"] is not None:
        parse_success += 1
        f1_scores.append(field_f1(r["pred_dict"], r["gt_dict"])["f1"])
        ted_scores.append(tree_edit_similarity(r["pred_dict"], r["gt_dict"]))
    else:
        f1_scores.append(0.0)
        ted_scores.append(0.0)

print(f"Field F1: {np.mean(f1_scores):.4f}")
print(f"Tree Edit Similarity: {np.mean(ted_scores):.4f}")
print(f"Parse Rate: {parse_success}/{len(sft_results)} ({parse_success/len(sft_results)*100:.1f}%)")

# ============================================================
# TODO #3 Solution — 錯誤分析
# ============================================================

indexed_scores = list(enumerate(f1_scores))
errors_sorted = sorted(indexed_scores, key=lambda x: x[1])

for rank, (idx, f1) in enumerate(errors_sorted[:5]):
    r = sft_results[idx]
    print(f"\n#{rank+1} | idx={idx} | F1={f1:.4f}")

    if r["pred_dict"] is not None:
        gt_flat = flatten_dict(r["gt_dict"])
        pred_flat = flatten_dict(r["pred_dict"])
        missing = [k for k in gt_flat if k not in pred_flat]
        wrong = [k for k in gt_flat if k in pred_flat and gt_flat[k] != pred_flat[k]]
        extra = [k for k in pred_flat if k not in gt_flat]
        print(f"  Missing: {len(missing)}, Wrong: {len(wrong)}, Extra: {len(extra)}")
    else:
        print(f"  JSON parse failed: {r['pred_text'][:100]}")
