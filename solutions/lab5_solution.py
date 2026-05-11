"""
Lab 5 Solution: 推論加速 + 部署 (Bonus)
=========================================
此為完整解答，供助教參考或學員對照。
"""

import time

import gradio as gr
import torch
from datasets import load_dataset
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_DIR = "./qwen25vl-cord-lora"
MERGED_DIR = "./qwen25vl-cord-merged"

SYSTEM_PROMPT = "你是專業的收據資訊抽取助手。請從圖片中抽取所有結構化資訊，以 JSON 格式輸出。"
INSTRUCTION = "請從這張收據圖片中抽取所有資訊，包含店名、地址、電話、日期、品項、價格、總計等，以 JSON 格式輸出。"

# 載入 + 合併
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()
model.eval()

# 保存
model.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)

# ============================================================
# TODO #1 Solution — Benchmark
# ============================================================

ds = load_dataset("naver-clova-ix/cord-v2")
test_image = ds["test"][0]["image"]


def benchmark_inference(model, processor, image, n_runs=5, warmup=2):
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

    # ✅ SOLUTION: Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    # ✅ SOLUTION: Benchmark
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
        total_tokens += len(output_ids[0]) - len(inputs["input_ids"][0])

    avg_time = sum(times) / len(times)
    avg_tokens = total_tokens / n_runs
    tokens_per_sec = avg_tokens / avg_time

    return {"avg_time": avg_time, "avg_tokens": avg_tokens, "tokens_per_sec": tokens_per_sec}


result = benchmark_inference(model, processor, test_image)
print(f"Avg time: {result['avg_time']:.2f}s | {result['tokens_per_sec']:.1f} tok/s")

# ============================================================
# TODO #2 Solution — Gradio Demo
# ============================================================


def process_receipt(image):
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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=768, do_sample=False)

    generated_ids = output_ids[0][len(inputs["input_ids"][0]):]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True)


# ✅ SOLUTION: Gradio 介面
demo = gr.Interface(
    fn=process_receipt,
    inputs=gr.Image(type="pil", label="上傳收據圖片"),
    outputs=gr.Textbox(label="抽取結果 (JSON)", lines=20),
    title="收據資訊抽取 Demo — Qwen2.5-VL-3B (SFT)",
    description="上傳一張收據圖片，經過 LoRA 微調的 Qwen2.5-VL-3B 將自動抽取結構化 JSON。",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
