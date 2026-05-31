# E-commerce Receipt Information Extraction via LLM Fine-Tuning

本專案為國立陽明交通大學「AI 演算法實務課程」之階段性實務專案。目標是利用大語言模型（LLM/VLM）將電商非結構化的原始收據文字，透過監督式微調（SFT）自動化轉化為結構化的 JSON 格式數據，解決電商後台自動化對帳與財務審核的痛點。

## 📌 專案核心架構 (Project Architecture)
本專案專注於**資料前處理 Pipeline 的建置**與**基準模型（Baseline）的資料格式化處理**：
1. **Data Preprocessing & Tokenization:** 將亂序、非結構化的電商收據欄位進行清洗，編製成 `receipt_words_list.json`。
2. **Dataset Split:** 嚴格執行機器學習實務標準，將資料集切分為 `train.json`（訓練集）與 `val.json`（驗證集），以防模型過擬合（Overfitting）。
3. **Model Configuration:** 基於 `Qwen2.5` 系列輕量化模型架構，建置文字端到端（Text-to-Text）的資訊抽取微調管線（SFT Pipeline）。

## 📊 資料格式範例 (Data Format)
- **Input (原始非結構化文字):** 收據內含商店名稱、交易日期、多個品項品名與個別價格。
- **Output (期望之結構化 JSON):**
```json
  {
    "store_name": "...",
    "date": "2026-05-31",
    "items": [{"name": "...", "price": 0}],
    "total_amount": 0
  }
