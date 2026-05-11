"""
Lab 0: 環境建置與驗證
====================
確認 GPU、CUDA、PyTorch 及所有依賴套件已正確安裝。

執行方式:
    uv run python lab0_setup.py
"""


def verify_environment():
    """驗證所有必要套件與硬體環境"""
    print("=" * 60)
    print("VLM SFT Lab — 環境驗證")
    print("=" * 60)

    # --- PyTorch + CUDA ---
    print("\n[1/5] PyTorch + CUDA")
    print("-" * 40)
    import torch

    print(f"  PyTorch version:  {torch.__version__}")
    print(f"  CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version:     {torch.version.cuda}")
        print(f"  GPU:              {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM:             {vram:.1f} GB")
        if vram < 15:
            print("  ⚠️  VRAM < 15GB，訓練時可能需要降低 batch_size 或 max_pixels")
    else:
        print("  ❌ CUDA 不可用！請檢查驅動程式版本。")

    # --- Transformers + Model support ---
    print("\n[2/5] Transformers + VLM 支援")
    print("-" * 40)
    import transformers

    print(f"  transformers:     {transformers.__version__}")
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401

    print("  Qwen2.5-VL:      ✅ 可載入")

    # --- Training stack ---
    print("\n[3/5] 訓練套件")
    print("-" * 40)
    import accelerate
    import peft
    import trl

    print(f"  accelerate:       {accelerate.__version__}")
    print(f"  peft:             {peft.__version__}")
    print(f"  trl:              {trl.__version__}")

    import bitsandbytes as bnb

    print(f"  bitsandbytes:     {bnb.__version__}")

    # --- Data & evaluation ---
    print("\n[4/5] 資料與評估套件")
    print("-" * 40)
    import datasets

    print(f"  datasets:         {datasets.__version__}")

    import PIL

    print(f"  Pillow:           {PIL.__version__}")

    import edit_distance  # noqa: F401

    print("  edit-distance:    ✅")

    import qwen_vl_utils  # noqa: F401

    print("  qwen-vl-utils:    ✅")

    import matplotlib

    print(f"  matplotlib:       {matplotlib.__version__}")

    # --- Quick GPU stress test ---
    print("\n[5/5] GPU 快速壓力測試")
    print("-" * 40)
    if torch.cuda.is_available():
        x = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
        y = x @ x.T
        del x, y
        torch.cuda.empty_cache()
        print("  bf16 matmul:      ✅ 正常")
        allocated = torch.cuda.memory_allocated() / 1e6
        print(f"  清理後 VRAM:      {allocated:.1f} MB allocated")
    else:
        print("  ⏭️  跳過（無 GPU）")

    print("\n" + "=" * 60)
    print("✅ 環境驗證完成！可以開始 Lab 1。")
    print("=" * 60)


if __name__ == "__main__":
    verify_environment()
