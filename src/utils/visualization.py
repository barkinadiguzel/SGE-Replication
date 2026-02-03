import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_group_maps(lengths, groups=8):
    if isinstance(lengths, torch.Tensor):
        lengths = lengths.detach().cpu().numpy()

    sample = lengths[0]
    G = sample.shape[0]
    idxs = np.linspace(0, G - 1, groups, dtype=int)

    plt.figure(figsize=(groups * 2, 2.5))
    for i, g in enumerate(idxs):
        ax = plt.subplot(1, groups, i + 1)
        m = sample[g]
        m = (m - m.min()) / (m.max() - m.min() + 1e-6)
        ax.imshow(m)
        ax.set_title(f"G{g}")
        ax.axis("off")
    plt.show()


def plot_histogram(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()

    plt.figure(figsize=(5, 3))
    plt.hist(values.ravel(), bins=60)
    plt.xlabel("activation")
    plt.ylabel("count")
    plt.show()
