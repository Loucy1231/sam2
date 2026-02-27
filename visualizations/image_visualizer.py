# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
图像分割可视化模块

提供基于 SAM2 的图像分割结果可视化功能，支持：
- 原始图像与分割掩码并排显示
- 彩色掩码叠加显示
- 分割边界绘制
- 多对象多掩码综合显示
- 将可视化结果保存到文件
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualizations.utils import (
    apply_colormap,
    create_color_palette,
    draw_boundaries,
    overlay_mask_on_image,
    show_box,
    show_mask,
    show_points,
)


class ImageVisualizer:
    """
    图像分割结果可视化器。

    使用示例：
        from visualizations import ImageVisualizer

        viz = ImageVisualizer(alpha=0.5, show_borders=True)
        viz.show_segmentation(image, masks, scores)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        show_borders: bool = True,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
    ) -> None:
        """
        初始化图像可视化器。

        参数：
            alpha: 掩码叠加透明度，范围 [0, 1]
            show_borders: 是否绘制掩码边界
            figsize: 图形尺寸（宽, 高），单位英寸
            dpi: 图形分辨率
        """
        self.alpha = alpha
        self.show_borders = show_borders
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # 公共可视化方法
    # ------------------------------------------------------------------

    def show_segmentation(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: Optional[np.ndarray] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        title: str = "SAM2 图像分割结果",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        显示所有预测掩码，每个掩码单独一个子图。

        参数：
            image: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks: 形状为 (N, H, W) 或 (N, 1, H, W) 的二值掩码数组
            scores: 长度为 N 的置信度分数数组（可选）
            point_coords: 形状为 (M, 2) 的提示点坐标（可选）
            point_labels: 长度为 M 的提示点标签（可选）
            box: 长度为 4 的边界框 [x0, y0, x1, y1]（可选）
            title: 整体标题
            save_path: 保存路径；为 None 时不保存

        返回：
            matplotlib Figure 对象
        """
        n_masks = len(masks)
        ncols = min(n_masks, 3)
        nrows = (n_masks + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(self.figsize[0] * ncols, self.figsize[1] * nrows))
        fig.suptitle(title, fontsize=16)

        # 统一成可迭代的轴列表
        if n_masks == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = [ax for row in axes for ax in row]

        for i in range(nrows * ncols):
            ax = axes[i]
            ax.axis("off")
            if i >= n_masks:
                continue

            ax.imshow(image)
            show_mask(masks[i], ax, alpha=self.alpha, borders=self.show_borders)

            if point_coords is not None and point_labels is not None:
                show_points(point_coords, point_labels, ax)
            if box is not None:
                show_box(box, ax)

            score_str = f"\n置信度：{scores[i]:.3f}" if scores is not None else ""
            ax.set_title(f"掩码 {i + 1}{score_str}", fontsize=12)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def show_best_mask(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        title: str = "最佳分割掩码",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        仅显示得分最高的掩码。

        参数：
            image: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks: 形状为 (N, H, W) 的掩码数组
            scores: 长度为 N 的置信度分数
            point_coords: 提示点坐标（可选）
            point_labels: 提示点标签（可选）
            box: 边界框（可选）
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        best_idx = int(np.argmax(scores))
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.imshow(image)
        show_mask(masks[best_idx], ax, alpha=self.alpha, borders=self.show_borders)
        if point_coords is not None and point_labels is not None:
            show_points(point_coords, point_labels, ax)
        if box is not None:
            show_box(box, ax)
        ax.set_title(f"{title}（置信度：{scores[best_idx]:.3f}）", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def show_comparison(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        scores: Optional[np.ndarray] = None,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        title: str = "分割结果对比",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        并排显示原始图像、掩码叠加图像和边界图像。

        参数：
            image: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks: 形状为 (N, H, W) 的掩码数组，取最佳掩码
            scores: 置信度分数（可选，用于选最佳掩码）
            point_coords: 提示点坐标（可选）
            point_labels: 提示点标签（可选）
            box: 边界框（可选）
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        if scores is not None:
            best_mask = masks[int(np.argmax(scores))]
        else:
            best_mask = masks[0]

        overlaid = overlay_mask_on_image(image, best_mask, alpha=self.alpha)
        boundary = draw_boundaries(image, best_mask)

        fig, axes = plt.subplots(1, 3, figsize=(self.figsize[0] * 3, self.figsize[1]))
        fig.suptitle(title, fontsize=16)

        axes[0].imshow(image)
        axes[0].set_title("原始图像", fontsize=13)
        axes[0].axis("off")
        if point_coords is not None and point_labels is not None:
            show_points(point_coords, point_labels, axes[0])
        if box is not None:
            show_box(box, axes[0])

        axes[1].imshow(overlaid)
        axes[1].set_title("掩码叠加", fontsize=13)
        axes[1].axis("off")

        axes[2].imshow(boundary)
        axes[2].set_title("分割边界", fontsize=13)
        axes[2].axis("off")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def show_multi_object(
        self,
        image: np.ndarray,
        masks_dict: Dict[int, np.ndarray],
        title: str = "多目标分割结果",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        在同一张图上叠加显示多个目标的分割掩码。

        参数：
            image: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks_dict: 字典，键为目标 ID，值为形状 (H, W) 的掩码
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        palette = create_color_palette(len(masks_dict))
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax.imshow(image)

        for idx, (obj_id, mask) in enumerate(masks_dict.items()):
            color = palette[idx % len(palette)]
            show_mask(mask, ax, color=color, alpha=self.alpha, borders=self.show_borders)

        ax.set_title(title, fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def save_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        save_path: Union[str, Path],
        scores: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        将掩码叠加图像保存为 PNG 文件并返回叠加结果。

        参数：
            image: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks: 形状为 (N, H, W) 的掩码数组
            save_path: 输出文件路径
            scores: 置信度分数（可选，用于选最佳掩码）
            alpha: 透明度；为 None 时使用初始化参数

        返回：
            叠加后的 uint8 RGB 图像
        """
        if alpha is None:
            alpha = self.alpha

        palette = create_color_palette(len(masks))
        result = image.copy()
        for i, mask in enumerate(masks):
            color = palette[i % len(palette)]
            result = overlay_mask_on_image(result, mask, color=color.tolist(), alpha=alpha)

        Image.fromarray(result).save(save_path)
        return result
