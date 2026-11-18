"""环境自检：确认 Python / PyTorch / GPU 状态，并给出建议配置。

用法：python code/check_env.py
"""
import platform
import sys
import time


def main() -> None:
    print("=" * 60)
    print("环境自检")
    print("=" * 60)
    print(f"Python      : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")

    try:
        import torch
    except ImportError:
        print("\n[错误] 没有安装 PyTorch。请先运行 run_windows.bat")
        sys.exit(1)

    print(f"PyTorch     : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用   : {cuda_ok}")

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        print(f"显卡        : {name}")
        print(f"显存        : {vram:.1f} GB")
        if vram >= 15:
            batch, amps = 64, True
        elif vram >= 9:
            batch, amps = 32, True
        else:
            batch, amps = 16, True
        print(f"建议参数    : --batch-size {batch}    AMP: 开")
    else:
        print("显卡        : 未检测到可用 CUDA")
        print("建议参数    : 只能用 CPU，速度约为 GPU 的 1/50，")
        print("              建议先跑 QUICK 模式验证管线，正式实验请用带独显的机器或租卡。")
        batch, amps = 8, False

    # 速度实测（不需要下载模型，用随机张量近似）
    import torch.nn as nn

    dev = "cuda" if cuda_ok else "cpu"
    net = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    ).to(dev)
    x = torch.randn(batch, 3, 224, 224, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    for _ in range(2):  # 预热
        opt.zero_grad(); nn.functional.cross_entropy(net(x), y).backward(); opt.step()
    t = time.time()
    for _ in range(5):
        opt.zero_grad(); nn.functional.cross_entropy(net(x), y).backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t) / 5
    print(f"\n参考吞吐    : 每秒约 {batch / dt:.0f} 张图（小型 CNN，仅用于比较机器快慢）")
    print("\n自检通过。下一步：")
    print("  1) cd code && python -m src.smoke_test")
    print("  2) 运行 run_windows.bat 里的 QUICK 模式跑一遍真实数据")


if __name__ == "__main__":
    main()
