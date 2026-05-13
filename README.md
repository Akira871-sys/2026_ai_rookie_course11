# VLM SFT Lab — Qwen2.5-VL-3B 收據資訊抽取

> 訓練 Vision LLM 從印尼收據（CORD-v2）抽取結構化 JSON

## 環境需求
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

## W&B (Weights & Biases) 訓練監控設定

本課程使用 [Weights & Biases](https://wandb.ai) 來即時視覺化訓練過程（loss 曲線、學習率、VRAM 使用量等）以及記錄評估結果。**請在執行 Lab 3 之前完成以下設定。**

### Step 1: 註冊帳號

前往 [https://wandb.ai/site](https://wandb.ai/site) 點擊 **Sign Up**，可使用 GitHub / Google 帳號快速註冊。

### Step 2: 取得 API Key

登入後前往 [https://wandb.ai/authorize](https://wandb.ai/authorize)，複製你的 API Key。

### Step 3: 設定環境變數

在終端機中執行 `wandb login`，貼上你的 API Key：

```bash
uv run wandb login
# 貼上你的 API Key（不會顯示在螢幕上），按 Enter
```

或者直接設定環境變數（不需要互動輸入）：

```bash
# Linux / macOS
export WANDB_API_KEY="你的_API_Key"

# Windows PowerShell
$env:WANDB_API_KEY="你的_API_Key"
```

### Step 4: 確認連線

```bash
uv run python -c "import wandb; wandb.init(project='test'); wandb.finish(); print('W&B 連線成功！')"
```

成功後你可以在 [https://wandb.ai](https://wandb.ai) 看到一個 `test` 專案。

### 在 Lab 中的用途

| Lab | W&B 功能 |
|-----|----------|
| Lab 3 | 即時追蹤 train loss、eval loss、learning rate 曲線 |
| Lab 4 | 記錄評估指標（F1、TED、Parse Rate）、上傳圖表與預測結果表格 |

> **離線模式**: 如果無法連網，可設定 `WANDB_MODE=offline`，訓練結束後再用 `wandb sync` 上傳。

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

1. 訓練結果 metrics 報告（zero-shot vs SFT 對照表）
2. W&B Dashboard 截圖（包含 loss 曲線與評估指標）
3. (選做) Push adapter 到 HuggingFace Hub
