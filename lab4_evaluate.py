"""
Lab 4: 評估 + 結果分析
=======================
學習目標:
  - 用結構化 metrics 評估 SFT 效果
  - 做 zero-shot vs SFT 的完整對照
  - 用 Tree Edit Distance 作為進階 metric
  - 系統性錯誤分析，找出失敗 pattern
  - 將評估指標與圖表上傳至 W&B

執行方式:
    uv run python lab4_evaluate.py

前置要求:
  - Lab 1 的 baseline_results.json
  - Lab 3 訓練好的 adapter (./qwen25vl-cord-lora/)
  - 已完成 wandb login（見 README.md）
"""

import json
import os

import edit_distance
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm.auto import tqdm

# ============================================================
# Section 1: 初始化 Weights & Biases
# ============================================================

wandb.init(
    project="vlm-sft-lab",
    name="qwen25vl-cord-eval",
    config={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "task": "receipt-extraction-eval",
        "dataset": "naver-clova-ix/cord-v2",
    },
)
print("W&B 初始化完成！評估結果將同步至 https://wandb.ai")

# ============================================================
# Section 2: 載入模型 + Adapter
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = "./qwen25vl-cord-lora"
NUM_EVAL_SAMPLES = 100

print("載入 base model + LoRA adapter...")
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
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
print("模型載入完成（含 LoRA adapter）。")

# 載入資料集
ds = load_dataset("naver-clova-ix/cord-v2")
test_ds = ds["test"]
print(f"Test set: {len(test_ds)} 筆")

# ============================================================
# Section 3: 推論函式
# ============================================================

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"


def predict_single(image, model, processor, max_new_tokens=768):
    """對單張收據做 SFT 推論"""
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

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, return_tensors="pt", padding=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)


def parse_json_safe(text):
    """安全地從模型輸出解析 JSON"""
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
# Section 4: TODO #1 — Batch 推論 🟡
# ============================================================


def batch_predict(model, processor, dataset, n_samples=NUM_EVAL_SAMPLES):
    """
    對 test set 做批次推論。

    Args:
        model: 載入 adapter 的模型
        processor: processor
        dataset: test dataset
        n_samples: 評估幾筆

    Returns:
        list of dict: [{"pred_text": str, "pred_dict": dict|None, "gt_dict": dict}, ...]
    """
    results = []

    for i in tqdm(range(min(n_samples, len(dataset))), desc="SFT 推論"):
        example = dataset[i]
        image = example["image"]
        gt = json.loads(example["ground_truth"])["gt_parse"]

        # TODO 1.1: 呼叫 predict_single 取得模型輸出 🟢
        pred_text = ___

        # TODO 1.2: 嘗試 parse JSON 🟢
        pred_dict = ___

        results.append({
            "index": i,
            "pred_text": pred_text,
            "pred_dict": pred_dict,
            "gt_dict": gt,
        })

    return results


print("\n開始 SFT 推論...")
sft_results = batch_predict(model, processor, test_ds)
print(f"完成！共 {len(sft_results)} 筆")

# ============================================================
# Section 5: 評估 Metrics
# ============================================================


def flatten_dict(d, prefix=""):
    """把 nested dict 攤平"""
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
    """計算 field-level F1"""
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
# Section 6: TODO #2 — Tree Edit Distance (TED) 🟡
# ============================================================


def normalize_tree(d):
    """
    遞迴標準化 dict/list:
    - dict: key 排序
    - list: 保持原順序
    - str: strip + lowercase
    """
    # TODO 2.1: 遞迴標準化 🟡
    # Hint:
    #   - dict → {k: normalize_tree(v) for k, v in sorted(d.items())}
    #   - list → [normalize_tree(item) for item in d]
    #   - str → d.strip().lower()
    #   - else → d
    if isinstance(d, dict):
        return ___
    elif isinstance(d, list):
        return ___
    elif isinstance(d, str):
        return ___
    else:
        return d


def tree_edit_similarity(pred, gt):
    """
    計算正規化 Tree Edit Similarity (0=完全錯, 1=完美)。

    做法: 把 tree 序列化成 sorted JSON string，計算字元層級的編輯距離，
    再正規化到 [0, 1]。
    """
    pred_str = json.dumps(normalize_tree(pred), sort_keys=True, ensure_ascii=False)
    gt_str = json.dumps(normalize_tree(gt), sort_keys=True, ensure_ascii=False)

    # TODO 2.2: 用 edit_distance 計算，並正規化 🟢
    # Hint: edit_distance.SequenceMatcher(a=pred_str, b=gt_str)
    #        sm.distance() 是編輯距離
    #        正規化: distance / max(len(pred_str), len(gt_str))
    sm = edit_distance.SequenceMatcher(a=pred_str, b=gt_str)
    distance = sm.distance()
    max_len = max(len(pred_str), len(gt_str))
    normalized_distance = ___

    return 1 - normalized_distance  # similarity: 越高越好


# ============================================================
# Section 7: 計算所有 metrics
# ============================================================

print("\n計算評估指標...")

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

# 讀取 baseline 結果
baseline_path = "baseline_results.json"
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
else:
    print("⚠️  找不到 baseline_results.json，請先執行 Lab 1")
    baseline = {"f1": 0.35, "parse_rate": 0.60}

# 輸出對照表
sft_f1 = np.mean(f1_scores)
sft_ted = np.mean(ted_scores)
sft_parse = parse_success / len(sft_results)

print("\n" + "=" * 60)
print("              Zero-shot vs SFT 對照表")
print("=" * 60)
print(f"{'Metric':<25} {'Zero-shot':<12} {'SFT':<12} {'改善':<12}")
print("-" * 60)
print(f"{'Field F1':<25} {baseline['f1']:<12.4f} {sft_f1:<12.4f} {sft_f1 - baseline['f1']:+.4f}")
print(f"{'Tree Edit Similarity':<25} {'~0.42':<12} {sft_ted:<12.4f} {'':<12}")
print(f"{'JSON Parse Rate':<25} {baseline['parse_rate']*100:<11.1f}% {sft_parse*100:<11.1f}% {(sft_parse - baseline['parse_rate'])*100:+.1f}%")
print("=" * 60)

# ============================================================
# Section 8: 視覺化
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart: 指標對照
metrics_names = ["Field F1", "Tree Edit Sim", "Parse Rate"]
zero_shot_vals = [baseline["f1"], 0.42, baseline["parse_rate"]]
sft_vals = [sft_f1, sft_ted, sft_parse]

x = np.arange(len(metrics_names))
width = 0.35
axes[0].bar(x - width / 2, zero_shot_vals, width, label="Zero-shot", color="skyblue")
axes[0].bar(x + width / 2, sft_vals, width, label="SFT", color="coral")
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names)
axes[0].set_ylim(0, 1)
axes[0].set_title("Zero-shot vs SFT")
axes[0].legend()
axes[0].set_ylabel("Score")

# F1 分佈直方圖
axes[1].hist(f1_scores, bins=20, edgecolor="black", alpha=0.7, color="coral")
axes[1].axvline(np.mean(f1_scores), color="red", linestyle="--", label=f"Mean: {np.mean(f1_scores):.3f}")
axes[1].set_xlabel("Field F1")
axes[1].set_ylabel("Count")
axes[1].set_title("SFT Field F1 Distribution")
axes[1].legend()

# TED 分佈
axes[2].hist(ted_scores, bins=20, edgecolor="black", alpha=0.7, color="mediumseagreen")
axes[2].axvline(np.mean(ted_scores), color="darkgreen", linestyle="--", label=f"Mean: {np.mean(ted_scores):.3f}")
axes[2].set_xlabel("Tree Edit Similarity")
axes[2].set_ylabel("Count")
axes[2].set_title("SFT Tree Edit Similarity Distribution")
axes[2].legend()

plt.tight_layout()
plt.savefig("lab4_evaluation.png", dpi=100)
plt.show()
print("\n圖表已存為 lab4_evaluation.png")

# ============================================================
# Section 9: TODO #3 — 錯誤分析 🔴
# ============================================================

print("\n" + "=" * 60)
print("錯誤分析: Top 5 失敗案例")
print("=" * 60)

# TODO 3.1: 找出 F1 最低的 5 筆 🟢
# Hint: 把 index 和 f1 配對，排序找最小的
indexed_scores = list(enumerate(f1_scores))
errors_sorted = sorted(indexed_scores, key=lambda x: x[1])

for rank, (idx, f1) in enumerate(errors_sorted[:5]):
    r = sft_results[idx]
    print(f"\n{'─' * 50}")
    print(f"#{rank+1} | Test sample {idx} | F1 = {f1:.4f}")
    print(f"{'─' * 50}")

    # TODO 3.2: 顯示對照資訊 🟡
    # 顯示: GT (前 200 字)、Prediction (前 200 字)
    # 如果 pred_dict 存在，找出哪些 key 錯了

    gt_str = json.dumps(r["gt_dict"], ensure_ascii=False, indent=2)
    print(f"  GT (前 200 字):")
    print(f"    {gt_str[:200]}")

    if r["pred_dict"] is not None:
        pred_str = json.dumps(r["pred_dict"], ensure_ascii=False, indent=2)
        print(f"  Pred (前 200 字):")
        print(f"    {pred_str[:200]}")

        # TODO 3.3 (Challenge 🔴): 找出具體哪些欄位預測錯了
        # Hint: flatten 兩邊，比較差異
        gt_flat = flatten_dict(r["gt_dict"])
        pred_flat = flatten_dict(r["pred_dict"])

        missing_keys = [k for k in gt_flat if k not in pred_flat]
        wrong_value_keys = [
            k for k in gt_flat
            if k in pred_flat and gt_flat[k] != pred_flat[k]
        ]
        extra_keys = [k for k in pred_flat if k not in gt_flat]

        if missing_keys:
            print(f"  缺失欄位 ({len(missing_keys)}): {missing_keys[:5]}")
        if wrong_value_keys:
            print(f"  值錯誤 ({len(wrong_value_keys)}): {wrong_value_keys[:5]}")
        if extra_keys:
            print(f"  多餘欄位 ({len(extra_keys)}): {extra_keys[:5]}")
    else:
        print(f"  Pred (JSON parse 失敗):")
        print(f"    {r['pred_text'][:200]}")

# 視覺化錯誤案例圖片
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for rank, (idx, f1) in enumerate(errors_sorted[:5]):
    image = test_ds[idx]["image"]
    axes[rank].imshow(image)
    axes[rank].set_title(f"#{rank+1} F1={f1:.2f}")
    axes[rank].axis("off")
plt.suptitle("Top 5 Worst Predictions", fontsize=14)
plt.tight_layout()
plt.savefig("lab4_error_cases.png", dpi=100)
plt.show()
print("\n錯誤案例圖片已存為 lab4_error_cases.png")

# ============================================================
# Section 10: 存下完整評估結果
# ============================================================

eval_results = {
    "sft": {
        "field_f1": float(sft_f1),
        "tree_edit_similarity": float(sft_ted),
        "parse_rate": float(sft_parse),
    },
    "baseline": baseline,
    "per_sample_f1": f1_scores,
    "per_sample_ted": ted_scores,
}

with open("eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
print("\n完整評估結果已存為 eval_results.json")

# ============================================================
# Section 11: 上傳評估結果到 W&B
# ============================================================

wandb.log({
    "eval/sft_field_f1": float(sft_f1),
    "eval/sft_tree_edit_similarity": float(sft_ted),
    "eval/sft_parse_rate": float(sft_parse),
    "eval/baseline_field_f1": baseline["f1"],
    "eval/baseline_parse_rate": baseline["parse_rate"],
    "eval/f1_improvement": float(sft_f1 - baseline["f1"]),
})

wandb.log({
    "charts/evaluation": wandb.Image("lab4_evaluation.png"),
    "charts/error_cases": wandb.Image("lab4_error_cases.png"),
})

columns = ["index", "f1_score", "pred_text_preview", "gt_preview", "parse_ok"]
table_data = []
for i, r in enumerate(sft_results):
    gt_str = json.dumps(r["gt_dict"], ensure_ascii=False)[:200]
    table_data.append([
        r["index"],
        f1_scores[i],
        r["pred_text"][:200],
        gt_str,
        r["pred_dict"] is not None,
    ])
wandb.log({"eval/predictions_table": wandb.Table(columns=columns, data=table_data)})

wandb.finish()
print("W&B run 已結束。完整評估記錄可至 https://wandb.ai 查看。")
