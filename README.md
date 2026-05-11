# VLM SFT Lab — Qwen2.5-VL-3B 收據資訊抽取

> 訓練 Vision LLM 從印尼收據（CORD-v2）抽取結構化 JSON

## 環境需求

- Ubuntu + NVIDIA GPU (16GB VRAM)
- CUDA 13.0
- Python 3.11+
- uv 套件管理器

## 快速開始

```bash
# 1. 確認 GPU
nvidia-smi
nvcc --version  # 應顯示 release 13.0

# 2. 安裝依賴
uv sync

# 3. 驗證環境
uv run python lab0_setup.py
```

## Lab 結構

| Lab | 主題 | 預計時間 |
|-----|------|----------|
| Lab 0 | 環境建置與驗證 | 15 min |
| Lab 1 | 資料探索 + Zero-shot Baseline | 45 min |
| Lab 2 | 資料前處理 + 自定義 Collator | 60 min |
| Lab 3 | LoRA SFT 訓練 | 45 min (+ 30-45 min 訓練) |
| Lab 4 | 評估 + 結果分析 | 45 min |
| Lab 5 | 推論加速 + 部署 (Bonus) | 30 min |

## 檔案結構

```
vlm-sft-lab/
├── lab0_setup.py           # 環境驗證
├── lab1_baseline.py        # 資料探索 + Zero-shot
├── lab2_data_collator.py   # 資料前處理 + Collator
├── lab3_train.py           # LoRA SFT 訓練
├── lab4_evaluate.py        # 評估 + 分析
├── lab5_deploy.py          # 推論加速 + 部署 (Bonus)
├── utils/
│   └── metrics.py          # 共用評估函式
├── solutions/
│   ├── lab1_solution.py
│   ├── lab2_solution.py
│   ├── lab3_solution.py
│   ├── lab4_solution.py
│   └── lab5_solution.py
├── pyproject.toml
└── README.md
```

## TODO 難度標記

- 🟢 **填空 (Fill-in)** — 只缺 1-2 個 token
- 🟡 **半開放 (Half-open)** — 提供簽章，寫 5-15 行
- 🔴 **挑戰 (Challenge)** — 給目標和提示，自己寫
- ⚪ **Bonus** — 完全開放

## 預期成果

| Metric | Zero-shot | After SFT | 改善 |
|--------|-----------|-----------|------|
| Field F1 | ~0.35 | ~0.78 | +0.43 |
| Tree Edit Sim | ~0.42 | ~0.85 | +0.43 |
| JSON parse 成功率 | ~60% | >95% | +35% |

## 交付物

1. 完整的 5 個 .py 檔（填好所有 TODO）
2. 訓練結果 metrics 報告（zero-shot vs SFT 對照表）
3. 錯誤分析（至少 3 個失敗 pattern，200 字）
4. (選做) Push adapter 到 HuggingFace Hub
