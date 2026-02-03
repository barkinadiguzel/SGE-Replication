import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from src.models.sgenet_stub import SGENetStub
from src.utils.visualization import plot_feature_map


def run_demo(img_size=224):
    model = SGENetStub()
    model.eval()

    x = torch.randn(1, 3, img_size, img_size)

    with torch.no_grad():
        feats = model(x)

    for stage in ["stage3", "stage4"]:
        feat = feats[stage]

        print(f"{stage} shape:", feat.shape)

        plot_feature_map(
            feat,
            title=f"{stage} (After SGE)"
        )


if __name__ == "__main__":
    run_demo()
