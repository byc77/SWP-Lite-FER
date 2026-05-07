# profile_swp_litepp.py
import argparse
import time
import os
import numpy as np

import torch
import torch.nn as nn

from networks.SWP_cbpbn import SWPNet


def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] missing:", missing)
    print("[load] unexpected:", unexpected)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def conv2d_flops(module, inp, out):
    # inp[0]: (N, Cin, Hin, Win)
    # out: (N, Cout, Hout, Wout)
    x = inp[0]
    batch_size = x.shape[0]
    out_channels = out.shape[1]
    out_h = out.shape[2]
    out_w = out.shape[3]

    kernel_h, kernel_w = module.kernel_size
    in_channels = module.in_channels
    groups = module.groups

    # Multiply-add counted as 1 MAC here
    filters_per_channel = out_channels
    conv_per_position_flops = kernel_h * kernel_w * in_channels * filters_per_channel / groups
    active_elements_count = batch_size * out_h * out_w
    total_flops = conv_per_position_flops * active_elements_count

    if module.bias is not None:
        total_flops += out_channels * active_elements_count

    return int(total_flops)


def linear_flops(module, inp, out):
    x = inp[0]
    batch_size = x.shape[0] if x.dim() > 1 else 1
    total_flops = batch_size * module.in_features * module.out_features
    if module.bias is not None:
        total_flops += batch_size * module.out_features
    return int(total_flops)


def bn_flops(module, inp, out):
    # approximate BN cost as one operation per element
    return int(out.numel())


def relu_flops(module, inp, out):
    return int(out.numel())


def pool_flops(module, inp, out):
    return int(out.numel())


def compute_flops_by_hooks(model, img_size, device):
    flops = []

    def add_hooks(m):
        def hook(module, inp, out):
            if isinstance(module, nn.Conv2d):
                flops.append(conv2d_flops(module, inp, out))
            elif isinstance(module, nn.Linear):
                flops.append(linear_flops(module, inp, out))
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                flops.append(bn_flops(module, inp, out))
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.SiLU, nn.GELU)):
                flops.append(relu_flops(module, inp, out))
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                flops.append(pool_flops(module, inp, out))

        if isinstance(
            m,
            (
                nn.Conv2d,
                nn.Linear,
                nn.BatchNorm2d,
                nn.BatchNorm1d,
                nn.ReLU,
                nn.ReLU6,
                nn.Sigmoid,
                nn.SiLU,
                nn.GELU,
                nn.AdaptiveAvgPool2d,
                nn.AvgPool2d,
                nn.MaxPool2d,
            ),
        ):
            return m.register_forward_hook(hook)
        return None

    handles = []
    for m in model.modules():
        h = add_hooks(m)
        if h is not None:
            handles.append(h)

    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        _ = model(dummy)

    for h in handles:
        h.remove()

    return sum(flops)


def measure_latency(model, img_size, device, warmup=30, repeat=100, tta_flip=False):
    model.eval()

    times = []

    with torch.no_grad():
        for _ in range(warmup):
            if tta_flip:
                x = torch.randn(2, 3, img_size, img_size, device=device)
                _ = model(x)
            else:
                x = torch.randn(1, 3, img_size, img_size, device=device)
                _ = model(x)

        for _ in range(repeat):
            if tta_flip:
                x = torch.randn(2, 3, img_size, img_size, device=device)
            else:
                x = torch.randn(1, 3, img_size, img_size, device=device)

            if device == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    median_ms = float(np.median(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return mean_ms, median_ms, std_ms, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--out_txt", type=str, default="")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warning] CUDA is not available, fallback to CPU.")
        args.device = "cpu"

    device = args.device

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        print(f"[torch] num_threads = {torch.get_num_threads()}")

    print(f"[device] {device}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[img_size] {args.img_size}")

    model = SWPNet(num_classes=7, pretrained=False).to(device)

    # lazy init for GCG
    with torch.no_grad():
        _ = model(torch.zeros(1, 3, args.img_size, args.img_size, device=device))

    load_checkpoint(model, args.checkpoint, device)

    total_params, trainable_params = count_params(model)

    flops = compute_flops_by_hooks(model, args.img_size, device)

    mean_ms, median_ms, std_ms, fps = measure_latency(
        model,
        args.img_size,
        device,
        warmup=args.warmup,
        repeat=args.repeat,
        tta_flip=False,
    )

    mean_ms_tta, median_ms_tta, std_ms_tta, fps_tta = measure_latency(
        model,
        args.img_size,
        device,
        warmup=args.warmup,
        repeat=args.repeat,
        tta_flip=True,
    )

    lines = []
    lines.append("===== SWP-Lite++ Model Profile =====")
    lines.append(f"Checkpoint: {args.checkpoint}")
    lines.append(f"Device: {device}")
    lines.append(f"Input size: 3 x {args.img_size} x {args.img_size}")
    lines.append("")
    lines.append(f"Params: {total_params:,} ({total_params / 1e6:.4f} M)")
    lines.append(f"Trainable Params: {trainable_params:,} ({trainable_params / 1e6:.4f} M)")
    lines.append(f"FLOPs/MACs approx: {flops:,} ({flops / 1e9:.4f} G)")
    lines.append("")
    lines.append("[CPU/GPU Latency]")
    lines.append(f"Standard no-TTA mean:   {mean_ms:.2f} ms | median: {median_ms:.2f} ms | std: {std_ms:.2f} ms | FPS: {fps:.2f}")
    lines.append(f"Flip-TTA mean:          {mean_ms_tta:.2f} ms | median: {median_ms_tta:.2f} ms | std: {std_ms_tta:.2f} ms | FPS: {fps_tta:.2f}")
    lines.append("")
    lines.append("Note: FLOPs here are approximate hook-based MAC counts. Latency depends on CPU load, threads, and environment.")

    report = "\n".join(lines)
    print(report)

    if args.out_txt:
        out_dir = os.path.dirname(args.out_txt)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[saved] {args.out_txt}")


if __name__ == "__main__":
    main()