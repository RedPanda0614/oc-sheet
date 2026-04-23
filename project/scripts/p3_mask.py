# train/p3_mask.py
"""
Expression-local soft mask for P3 training.

Anime 脸布局固定，不依赖关键点检测，用固定比例划定眉眼/嘴巴区域。
mask 均值归一化为 1.0，保证总 loss 量级与 P1 相同，仅改变空间权重分布。
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
    构建表情区域软 mask。

    Anime 脸的固定比例（从上到下）：
      0%  - 15%  : 头发/额头        权重 1.0
      15% - 50%  : 眉毛 + 眼睛      权重 expr_weight
      50% - 58%  : 鼻子/脸颊        权重 1.0
      58% - 82%  : 嘴巴             权重 expr_weight
      82% - 100% : 下巴/颈部        权重 1.0

    Args:
        size:        图像边长（像素），假设正方形
        expr_weight: 表情区域相对于背景的权重倍数，默认 3.0
        sigma:       高斯模糊半径，使边界平滑，默认 30
        device:      返回 tensor 所在设备

    Returns:
        mask: shape (1, 1, size, size), float32
              均值归一化到 1.0，可直接与 loss 逐元素相乘
    """
    # 基础权重全 1
    mask = np.ones((size, size), dtype=np.float32)

    # 表情区域额外加权
    extra = np.zeros((size, size), dtype=np.float32)
    for top_ratio, bot_ratio in [(0.15, 0.50), (0.58, 0.82)]:
        top = int(size * top_ratio)
        bot = int(size * bot_ratio)
        extra[top:bot, :] = expr_weight - 1.0   # 加 (weight-1)，叠加到基础1上

    # 高斯平滑：让边界过渡自然，避免硬边界造成伪影
    extra = gaussian_filter(extra, sigma=sigma)

    mask = mask + extra

    # 均值归一化：保证 mask.mean() == 1.0
    # 这样 (loss * mask).mean() 的量级与 loss.mean() 相同
    mask = mask / mask.mean()

    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)


def masked_mse_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    加权 MSE loss，在表情区域给更高权重。

    Args:
        noise_pred: (B, C, H, W)  UNet 预测的噪声
        noise:      (B, C, H, W)  真实噪声
        mask:       (1, 1, H, W)  build_soft_mask 的输出，自动 broadcast

    Returns:
        scalar loss
    """
    # per-pixel MSE，然后乘以 mask 权重
    loss = ((noise_pred.float() - noise.float()) ** 2) * mask
    return loss.mean()


# ── 可视化工具（调试用，不影响训练）──────────────────────────────────

def visualize_mask(size: int = 512, expr_weight: float = 3.0, sigma: float = 30):
    """
    把 mask 保存为灰度图，方便确认区域是否正确。
    运行：python p3_mask.py
    """
    import os
    from PIL import Image

    mask = build_soft_mask(size, expr_weight, sigma).squeeze().numpy()

    # 归一化到 [0, 255] 显示
    mask_vis = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_vis = (mask_vis * 255).astype(np.uint8)

    out_path = "results/debug_p3_mask.png"
    os.makedirs("results", exist_ok=True)
    Image.fromarray(mask_vis, mode="L").save(out_path)
    print(f"Mask saved to {out_path}")
    print(f"  min={mask.min():.3f}  max={mask.max():.3f}  mean={mask.mean():.3f}")


if __name__ == "__main__":
    visualize_mask()
