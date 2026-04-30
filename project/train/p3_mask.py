# train/p3_mask.py
"""
Expression-local soft mask for P3 training.

Anime face layout is fixed; brow/eye and mouth regions are defined by fixed
proportions. The mask is mean-normalised to 1.0 so total loss magnitude
stays the same as P1, only the spatial weight distribution changes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def build_soft_mask(
    size: int = 512,
    expr_weight: float = 3.0,
    sigma: float = 30,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a soft spatial mask for expression regions.

    Fixed proportions (top to bottom):
      0%  - 15%  : hair/forehead     weight 1.0
      15% - 50%  : brows + eyes      weight expr_weight
      50% - 58%  : nose/cheeks       weight 1.0
      58% - 82%  : mouth             weight expr_weight
      82% - 100% : chin/neck         weight 1.0

    Returns:
        mask: shape (1, 1, size, size), float32, mean normalised to 1.0
    """
    mask = np.ones((size, size), dtype=np.float32)

    extra = np.zeros((size, size), dtype=np.float32)
    for top_ratio, bot_ratio in [(0.15, 0.50), (0.58, 0.82)]:
        top = int(size * top_ratio)
        bot = int(size * bot_ratio)
        extra[top:bot, :] = expr_weight - 1.0

    extra = gaussian_filter(extra, sigma=sigma)

    mask = mask + extra

    mask = mask / mask.mean()

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted MSE loss with higher weight on expression regions.

    Args:
        noise_pred: (B, C, H, W)
        noise:      (B, C, H, W)
        mask:       (1, 1, H, W) from build_soft_mask, auto-broadcast

    Returns:
        scalar loss
    """
    loss = ((noise_pred.float() - noise.float()) ** 2) * mask
    return loss.mean()


def visualize_mask(size: int = 512, expr_weight: float = 3.0, sigma: float = 30):
    """Save mask as a greyscale image for debugging. Run: python p3_mask.py"""
    import os
    from PIL import Image

    mask = build_soft_mask(size, expr_weight, sigma).squeeze().numpy()

    mask_vis = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_vis = (mask_vis * 255).astype(np.uint8)

    out_path = "results/debug_p3_mask.png"
    os.makedirs("results", exist_ok=True)
    Image.fromarray(mask_vis, mode="L").save(out_path)
    print(f"Mask saved to {out_path}")
    print(f"  min={mask.min():.3f}  max={mask.max():.3f}  mean={mask.mean():.3f}")


if __name__ == "__main__":
    visualize_mask()
