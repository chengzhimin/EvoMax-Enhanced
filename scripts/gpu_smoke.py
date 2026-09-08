import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.version.cuda}")
print(f"BF16: {torch.cuda.is_bf16_supported()}")
