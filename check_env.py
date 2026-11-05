"""PLACE 环境自检：确认依赖完整、GPU 可用。

用法：
    C:\\Users\\Huang\\anaconda3\\envs\\PLACE\\python.exe check_env.py
"""

import importlib
import platform
import sys
import time

ITEMS = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("cv2", "OpenCV"),
    ("ultralytics", "Ultralytics YOLO"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
    ("sklearn", "scikit-learn"),
    ("onnx", "ONNX"),
    ("onnxruntime", "ONNX Runtime"),
    ("lap", "LAP（ByteTrack 数据关联）"),
    ("filterpy", "FilterPy（卡尔曼滤波）"),
    ("albumentations", "Albumentations（数据增强）"),
    ("yaml", "PyYAML"),
    ("tensorboard", "TensorBoard"),
    ("tqdm", "tqdm"),
    ("psutil", "psutil"),
]


def main() -> int:
    print("=" * 62)
    print("PLACE 环境自检")
    print("=" * 62)
    print(f"Python      : {sys.version.split()[0]}  ({platform.machine()})")
    print(f"解释器路径  : {sys.executable}")
    print()

    missing = []
    for module, label in ITEMS:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "")
            print(f"  [OK]   {label:<28} {version}")
        except Exception as exc:  # noqa: BLE001
            missing.append(module)
            print(f"  [FAIL] {label:<28} {type(exc).__name__}: {exc}")

    print()
    print("-" * 62)
    print("GPU 检查")
    print("-" * 62)

    try:
        import torch

        print(f"  编译时 CUDA : {torch.version.cuda}")
        available = torch.cuda.is_available()
        print(f"  CUDA 可用   : {available}")

        if available:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  显卡        : {name}")
            print(f"  计算能力    : sm_{cap[0]}{cap[1]}")
            print(f"  显存        : {total:.1f} GB")

            a = torch.randn(4096, 4096, device="cuda")
            b = torch.randn(4096, 4096, device="cuda")
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                a @ b
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) / 10
            tflops = 2 * 4096**3 / elapsed / 1e12
            print(f"  实测算力    : {tflops:.1f} TFLOPS (fp32 矩阵乘)")
        else:
            print("  !! CUDA 不可用，需要检查 PyTorch 是否为 cu128 版本")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")

    print()
    print("-" * 62)
    if missing:
        print(f"结论：缺少 {len(missing)} 个包 -> {', '.join(missing)}")
        return 1
    print("结论：全部依赖就绪")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
