# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
中间张量可视化模块

提供 SAM2 模型内部中间张量的可视化功能，包括：
- Image Encoder 输出的特征图可视化
- Prompt Encoder 处理结果可视化
- Mask Decoder 各层输出可视化
- 注意力权重热力图
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualizations.utils import apply_colormap


class TensorVisualizer:
    """
    SAM2 模型中间张量可视化器。

    使用示例：
        from visualizations import TensorVisualizer

        tv = TensorVisualizer()
        tv.show_feature_maps(features, title="Image Encoder 特征图")
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 100,
        cmap: str = "jet",
    ) -> None:
        """
        初始化张量可视化器。

        参数：
            figsize: 图形尺寸（宽, 高），单位英寸
            dpi: 图形分辨率
            cmap: matplotlib 颜色映射名称
        """
        self.figsize = figsize
        self.dpi = dpi
        self.cmap = cmap

    # ------------------------------------------------------------------
    # 特征图可视化
    # ------------------------------------------------------------------

    def show_feature_maps(
        self,
        features: Union[torch.Tensor, np.ndarray],
        max_channels: int = 16,
        title: str = "特征图",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        可视化特征图的各通道。

        参数：
            features: 形状为 (C, H, W)、(1, C, H, W) 或 (B, C, H, W) 的张量/数组
            max_channels: 最多显示的通道数
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        feat = self._to_numpy(features)

        # 支持多种输入形状
        if feat.ndim == 4:
            feat = feat[0]  # 取第一个批次
        if feat.ndim == 2:
            feat = feat[np.newaxis]  # 添加通道维度

        n_channels = min(feat.shape[0], max_channels)
        ncols = min(n_channels, 4)
        nrows = (n_channels + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(self.figsize[0], self.figsize[1] * nrows // 2),
        )
        fig.suptitle(title, fontsize=15)

        axes_flat = np.array(axes).flatten() if n_channels > 1 else [axes]

        for i in range(len(axes_flat)):
            axes_flat[i].axis("off")
            if i >= n_channels:
                continue
            channel_map = feat[i]
            axes_flat[i].imshow(channel_map, cmap=self.cmap)
            axes_flat[i].set_title(f"通道 {i}", fontsize=10)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def show_feature_summary(
        self,
        features: Union[torch.Tensor, np.ndarray],
        title: str = "特征图汇总",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        显示特征图的统计汇总（均值图、最大值图、最小值图、标准差图）。

        参数：
            features: 形状为 (C, H, W) 或 (B, C, H, W) 的张量/数组
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        feat = self._to_numpy(features)
        if feat.ndim == 4:
            feat = feat[0]
        if feat.ndim == 2:
            feat = feat[np.newaxis]

        summaries = {
            "通道均值": feat.mean(axis=0),
            "通道最大值": feat.max(axis=0),
            "通道最小值": feat.min(axis=0),
            "通道标准差": feat.std(axis=0),
        }

        fig, axes = plt.subplots(1, 4, figsize=(self.figsize[0], self.figsize[1] // 2))
        fig.suptitle(title, fontsize=15)

        for ax, (name, data) in zip(axes, summaries.items()):
            im = ax.imshow(data, cmap=self.cmap)
            ax.set_title(name, fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 注意力权重可视化
    # ------------------------------------------------------------------

    def show_attention_map(
        self,
        attention: Union[torch.Tensor, np.ndarray],
        image: Optional[np.ndarray] = None,
        head_idx: int = 0,
        title: str = "注意力热力图",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        可视化注意力权重矩阵。

        参数：
            attention: 形状为 (H_q, H_k)、(num_heads, H_q, H_k) 或
                       (B, num_heads, H_q, H_k) 的注意力权重
            image: 原始图像（可选），用于在注意力图旁对比显示
            head_idx: 多头注意力中选取的头索引
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        attn = self._to_numpy(attention)

        if attn.ndim == 4:
            attn = attn[0, head_idx]
        elif attn.ndim == 3:
            attn = attn[head_idx]

        ncols = 3 if image is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(self.figsize[0] // ncols * ncols, self.figsize[1] // 2))
        fig.suptitle(title, fontsize=15)

        # 原始图像（可选）
        col = 0
        if image is not None:
            axes[col].imshow(image)
            axes[col].set_title("原始图像", fontsize=12)
            axes[col].axis("off")
            col += 1

        # 注意力矩阵热力图
        im = axes[col].imshow(attn, cmap="hot")
        axes[col].set_title(f"注意力矩阵（头 {head_idx}）", fontsize=12)
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)
        col += 1

        # 伪彩色注意力图
        colored = apply_colormap(attn)
        axes[col].imshow(colored)
        axes[col].set_title("伪彩色注意力", fontsize=12)
        axes[col].axis("off")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 掩码解码器输出可视化
    # ------------------------------------------------------------------

    def show_mask_logits(
        self,
        logits: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.0,
        title: str = "Mask Decoder 输出",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        可视化 Mask Decoder 的 logit 输出以及二值化掩码。

        参数：
            logits: 形状为 (N, H, W)、(B, N, H, W) 的掩码 logit 张量
            threshold: 二值化阈值
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        logit_np = self._to_numpy(logits)
        if logit_np.ndim == 4:
            logit_np = logit_np[0]

        n = logit_np.shape[0]
        fig, axes = plt.subplots(n, 2, figsize=(self.figsize[0] // 2 * 2, self.figsize[1] // 2 * n))
        fig.suptitle(title, fontsize=15)

        if n == 1:
            axes = [axes]

        for i in range(n):
            logit = logit_np[i]
            binary = (logit > threshold).astype(np.float32)

            axes[i][0].imshow(logit, cmap=self.cmap)
            axes[i][0].set_title(f"Logit {i}", fontsize=11)
            axes[i][0].axis("off")

            axes[i][1].imshow(binary, cmap="gray")
            axes[i][1].set_title(f"二值掩码 {i}（阈值={threshold}）", fontsize=11)
            axes[i][1].axis("off")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    def show_encoder_features_comparison(
        self,
        features_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
        title: str = "编码器特征图对比",
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        并排对比显示多个编码层的特征图均值。

        参数：
            features_dict: 字典，键为层名称，值为对应的特征张量
            title: 图形标题
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        n = len(features_dict)
        fig, axes = plt.subplots(1, n, figsize=(self.figsize[0] // n * n, self.figsize[1] // 2))
        fig.suptitle(title, fontsize=15)

        if n == 1:
            axes = [axes]

        for ax, (name, feat) in zip(axes, features_dict.items()):
            feat_np = self._to_numpy(feat)
            if feat_np.ndim == 4:
                feat_np = feat_np[0]
            if feat_np.ndim == 3:
                feat_np = feat_np.mean(axis=0)
            im = ax.imshow(feat_np, cmap=self.cmap)
            ax.set_title(name, fontsize=11)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """将 torch.Tensor 或 numpy 数组统一转换为 float32 numpy 数组。"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().float().numpy()
        return np.asarray(tensor, dtype=np.float32)
