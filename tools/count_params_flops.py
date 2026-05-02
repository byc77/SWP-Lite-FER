import torch
from thop import profile
from networks.SWP_cbpbn import SWPNet


def count_model(model_name: str, use_eca: bool, use_gcg: bool, img_size: int = 112):
    model = SWPNet(
        num_classes=7,
        pretrained=False,
        use_eca=use_eca,
        use_gcg=use_gcg
    )
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    # ??warm-up 銝甈∴?霈?lazy modules嚗?憒?GCG ??mlp嚗?撱箇?摰?
    with torch.no_grad():
        _ = model(dummy)

    flops, params = profile(model, inputs=(dummy,), verbose=False)

    print("=" * 60)
    print(f"Model      : {model_name}")
    print(f"use_eca    : {use_eca}")
    print(f"use_gcg    : {use_gcg}")
    print(f"Input size : 1 x 3 x {img_size} x {img_size}")
    print(f"Params     : {params:,} ({params / 1e6:.4f} M)")
    print(f"FLOPs      : {flops:,} ({flops / 1e9:.4f} G)")
    print("=" * 60)


if __name__ == "__main__":
    count_model(
        model_name="Baseline (SWPNet w/o ECA/GCG)",
        use_eca=False,
        use_gcg=False,
        img_size=112
    )

    count_model(
        model_name="Ours (SWP-Lite++)",
        use_eca=True,
        use_gcg=True,
        img_size=112
    )

