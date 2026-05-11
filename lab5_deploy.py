"""
Lab 5 (Bonus): 推論加速 + 部署
================================
學習目標:
  - 合併 LoRA adapter 到 base model
  - 比較 merged vs adapter-mode 推論速度
  - 了解進階推論加速選項 (vLLM, llama.cpp, ONNX)
  - 用 Gradio 做一個簡易 demo

執行方式:
    uv run python lab5_deploy.py

前置要求:
  - Lab 3 訓練好的 adapter (./qwen25vl-cord-lora/)
"""

import time

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ============================================================
# Section 1: 合併 LoRA (Merge & Unload)
# ============================================================

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = "./qwen25vl-cord-lora"
MERGED_DIR = "./qwen25vl-cord-merged"

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"

print("載入 base model + adapter...")
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
print("Adapter 載入完成。")

# 合併 LoRA 權重到 base model
print("\n合併 LoRA adapter 到 base model...")
model = model.merge_and_unload()
print("合併完成！現在是一個獨立模型，不再需要 adapter 檔案。")

# ============================================================
# Section 2: 保存完整合併模型
# ============================================================

print(f"\n保存合併後的模型到: {MERGED_DIR}")
model.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)

import os
merged_size = sum(
    os.path.getsize(os.path.join(dirpath, f))
    for dirpath, _, filenames in os.walk(MERGED_DIR)
    for f in filenames
)
print(f"  合併模型大小: {merged_size / 1e9:.2f} GB")
print("  (對比 adapter 只有 ~30MB，合併後是完整模型大小)")

# ============================================================
# Section 3: TODO #1 — 比較推論速度 🟡
# ============================================================


def benchmark_inference(model, processor, image, n_runs=5, warmup=2):
    """
    測量推論速度。

    Args:
        model: 模型
        processor: processor
        image: PIL Image
        n_runs: 正式測量次數
        warmup: 預熱次數（不計入統計）

    Returns:
        dict: {"avg_time": float, "tokens_per_sec": float}
    """
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

    # TODO 1.1: Warmup — 跑幾次讓 GPU 暖機（不計時）🟢
    for _ in range(warmup):
        with torch.no_grad():
            ___ = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # TODO 1.2: 正式測量 — 計算平均推論時間和 tokens/sec 🟡
    times = []
    total_tokens = 0

    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)

        # 計算生成的 token 數
        generated_len = len(output_ids[0]) - len(inputs["input_ids"][0])
        total_tokens += generated_len

    avg_time = sum(times) / len(times)
    avg_tokens = total_tokens / n_runs
    tokens_per_sec = avg_tokens / avg_time

    return {
        "avg_time": avg_time,
        "avg_tokens": avg_tokens,
        "tokens_per_sec": tokens_per_sec,
    }


# 測試用圖片
from datasets import load_dataset
ds = load_dataset("naver-clova-ix/cord-v2")
test_image = ds["test"][0]["image"]

print("\n" + "=" * 60)
print("推論速度 Benchmark (merged model)")
print("=" * 60)

result = benchmark_inference(model, processor, test_image)
print(f"  平均推論時間:  {result['avg_time']:.2f} sec")
print(f"  平均生成 tokens: {result['avg_tokens']:.0f}")
print(f"  生成速度:      {result['tokens_per_sec']:.1f} tokens/sec")

# ============================================================
# Section 4: 推論加速選項介紹
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  推論加速方案比較                                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. vLLM (生產推薦)                                         ║
║     - PagedAttention, continuous batching                    ║
║     - 2-4x throughput improvement                           ║
║     - pip install vllm                                      ║
║     - 支援 Qwen2.5-VL                                      ║
║                                                              ║
║  2. llama.cpp (邊緣設備)                                    ║
║     - GGUF 量化格式，CPU/GPU 混合推論                       ║
║     - Q4_K_M 量化後模型只需 ~2GB                            ║
║     - 適合離線/嵌入式場景                                   ║
║                                                              ║
║  3. ONNX Runtime (跨平台)                                   ║
║     - 支援 DirectML, TensorRT 後端                          ║
║     - 適合 Windows / 非 NVIDIA 硬體                         ║
║                                                              ║
║  4. TensorRT-LLM (極致 NVIDIA 優化)                         ║
║     - 需要額外編譯步驟                                      ║
║     - 最高 throughput，但部署複雜                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# ============================================================
# Section 5: TODO #2 — Gradio Demo ⚪ (Bonus)
# ============================================================

# 這個 TODO 是完全開放的 Bonus，學員可以回家自己完成。
# 以下提供骨架，學員可以自行擴充。


def create_gradio_demo(model, processor):
    """
    建立一個 Gradio 介面，讓使用者上傳收據圖片，
    模型即時抽取結構化 JSON。
    """
    import gradio as gr

    def process_receipt(image):
        """處理上傳的收據圖片"""
        if image is None:
            return "請上傳一張收據圖片"

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
            output_ids = model.generate(
                **inputs, max_new_tokens=768, do_sample=False
            )

        generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
        result = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result

    # TODO 2.1 (Bonus ⚪): 建立 Gradio 介面
    # Hint: gr.Interface 或 gr.Blocks
    demo = gr.Interface(
        fn=process_receipt,
        inputs=gr.Image(type="pil", label="上傳收據圖片"),
        outputs=gr.Textbox(label="抽取結果 (JSON)", lines=20),
        title="收據資訊抽取 Demo",
        description="上傳一張收據圖片，模型將自動抽取結構化資訊。",
        examples=[],  # TODO: 可以放幾張範例圖片路徑
    )

    return demo


# ============================================================
# Section 6: 啟動 Demo (取消註解即可使用)
# ============================================================

# 取消下面的註解來啟動 Gradio demo:
# demo = create_gradio_demo(model, processor)
# demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

print("""
╔══════════════════════════════════════════════════════════════╗
║  Lab 5 完成！                                               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  ☐ LoRA 成功合併到 base model                               ║
║  ☐ 合併後的模型可以正常推論                                 ║
║  ☐ 測量了推論速度 (tokens/sec)                              ║
║  ☐ 了解各種推論加速方案的適用場景                            ║
║  ☐ (Bonus) Gradio demo 可以跑                               ║
║                                                              ║
║  恭喜完成所有 Lab！🎉                                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

如果你想啟動 Gradio demo，取消 Section 6 的註解後重新執行:
    uv run python lab5_deploy.py

Gradio 會在 http://localhost:7860 提供 web 介面。
""")
