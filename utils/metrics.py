"""
共用評估工具
============
提供 Field F1、Tree Edit Similarity、JSON parse 等共用函式，
Lab 1 和 Lab 4 都會用到。
"""

import json

import edit_distance


def flatten_dict(d, prefix=""):
    """
    將 nested dict/list 攤平成 {path: value} 格式。

    Examples:
        {"menu": [{"nm": "Coffee", "price": "5000"}]}
        → {"menu.0.nm": "coffee", "menu.0.price": "5000"}
    """
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
    """
    計算 field-level F1。

    將 pred 和 gt 各自 flatten 後比較 key-value pairs。
    TP = 同 key 且同 value
    FP = pred 有但 gt 沒有，或 value 不同
    FN = gt 有但 pred 沒有

    Returns:
        dict with "precision", "recall", "f1"
    """
    pred_flat = flatten_dict(pred_dict)
    gt_flat = flatten_dict(gt_dict)

    tp = sum(1 for k in pred_flat if k in gt_flat and pred_flat[k] == gt_flat[k])
    fp = sum(1 for k in pred_flat if k not in gt_flat or pred_flat[k] != gt_flat[k])
    fn = sum(1 for k in gt_flat if k not in pred_flat or gt_flat[k] != pred_flat.get(k))

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_tree(d):
    """
    遞迴標準化 nested structure:
    - dict: key 按字母排序
    - list: 保持原順序
    - str: strip + lowercase
    """
    if isinstance(d, dict):
        return {k: normalize_tree(v) for k, v in sorted(d.items())}
    elif isinstance(d, list):
        return [normalize_tree(item) for item in d]
    elif isinstance(d, str):
        return d.strip().lower()
    else:
        return d


def tree_edit_similarity(pred, gt):
    """
    計算正規化 Tree Edit Similarity (0=完全錯, 1=完美)。

    將 tree 序列化成 sorted JSON string，
    用字元層級編輯距離衡量差異，正規化到 [0, 1]。
    """
    pred_str = json.dumps(normalize_tree(pred), sort_keys=True, ensure_ascii=False)
    gt_str = json.dumps(normalize_tree(gt), sort_keys=True, ensure_ascii=False)

    sm = edit_distance.SequenceMatcher(a=pred_str, b=gt_str)
    distance = sm.distance()
    max_len = max(len(pred_str), len(gt_str), 1)
    normalized_distance = distance / max_len

    return 1 - normalized_distance


def parse_json_safe(text):
    """
    嘗試從模型輸出中解析 JSON。
    處理 markdown code block 包裹的情況。

    Returns:
        dict or None
    """
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
