# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
通用工具函数模块

提供可视化相关的基础工具函数，包括颜色处理、掩码叠加、边界绘制等。
"""

from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


# 预定义颜色调色板（RGB格式）
DEFAULT_PALETTE = [
    [255, 0, 0],    # 红色
    [0, 255, 0],    # 绿色
    [0, 0, 255],    # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 洋红
    [0, 255, 255],  # 青色
    [255, 128, 0],  # 橙色
    [128, 0, 255],  # 紫色
    [0, 128, 255],  # 天蓝
    [255, 0, 128],  # 粉红
]


def create_color_palette(n_colors: int, seed: int = 42) -> np.ndarray:
    """
    创建包含指定数量颜色的调色板。

    参数：
        n_colors: 颜色数量
        seed: 随机种子，用于可重复生成颜色

    返回：
        形状为 (n_colors, 3) 的 uint8 numpy 数组，每行为 RGB 颜色值
    """
    if n_colors <= len(DEFAULT_PALETTE):
        return np.array(DEFAULT_PALETTE[:n_colors], dtype=np.uint8)

    rng = np.random.default_rng(seed)
    extra = rng.integers(0, 256, size=(n_colors - len(DEFAULT_PALETTE), 3), dtype=np.uint8)
    return np.vstack([np.array(DEFAULT_PALETTE, dtype=np.uint8), extra])


def show_mask(
    mask: np.ndarray,
    ax: plt.Axes,
    color: Optional[np.ndarray] = None,
    alpha: float = 0.6,
    random_color: bool = False,
    borders: bool = True,
) -> None:
    """
    在 matplotlib 轴上显示分割掩码。

    参数：
        mask: 形状为 (H, W) 或 (1, H, W) 的二值掩码
        ax: matplotlib 轴对象
        color: RGB 颜色，形状为 (3,) 的数组；为 None 时使用默认蓝色
        alpha: 掩码透明度，范围 [0, 1]
        random_color: 是否使用随机颜色
        borders: 是否绘制掩码边界
    """
    if random_color:
        rng = np.random.default_rng()
        rgb = rng.random(3)
    elif color is not None:
        rgb = np.asarray(color, dtype=float) / 255.0 if np.max(color) > 1 else np.asarray(color, dtype=float)
    else:
        rgb = np.array([30 / 255, 144 / 255, 255 / 255])

    rgba = np.append(rgb, alpha)

    mask_2d = mask.squeeze()
    h, w = mask_2d.shape[-2:]
    mask_uint8 = mask_2d.astype(np.uint8)
    mask_image = mask_uint8.reshape(h, w, 1) * rgba.reshape(1, 1, -1)

    if borders:
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)

    ax.imshow(mask_image)


def show_points(
    coords: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes,
    marker_size: int = 375,
) -> None:
    """
    在 matplotlib 轴上显示提示点（前景点为绿色星形，背景点为红色星形）。

    参数：
        coords: 形状为 (N, 2) 的点坐标数组，每行为 (x, y)
        labels: 长度为 N 的标签数组，1 表示前景，0 表示背景
        ax: matplotlib 轴对象
        marker_size: 标记大小
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1],
        color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1],
        color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25,
    )


def show_box(box: np.ndarray, ax: plt.Axes, color: str = "green", linewidth: int = 2) -> None:
    """
    在 matplotlib 轴上显示边界框。

    参数：
        box: 长度为 4 的数组，格式为 [x0, y0, x1, y1]
        ax: matplotlib 轴对象
        color: 边框颜色
        linewidth: 线宽
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=linewidth)
    )


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    将分割掩码叠加到图像上，返回合成后的 RGB 图像。

    参数：
        image: 形状为 (H, W, 3) 的 uint8 RGB 图像
        mask: 形状为 (H, W) 或 (1, H, W) 的二值掩码
        color: RGB 颜色列表或元组；为 None 时使用默认蓝色
        alpha: 掩码透明度，范围 [0, 1]

    返回：
        形状为 (H, W, 3) 的 uint8 合成图像
    """
    if color is None:
        color = [30, 144, 255]

    mask_2d = mask.squeeze().astype(bool)
    result = image.copy()
    color_arr = np.array(color, dtype=np.uint8)
    result[mask_2d] = (
        (1 - alpha) * result[mask_2d].astype(float) + alpha * color_arr.astype(float)
    ).astype(np.uint8)
    return result


def draw_boundaries(
    image: np.ndarray,
    mask: np.ndarray,
    color: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制掩码边界轮廓。

    参数：
        image: 形状为 (H, W, 3) 的 uint8 RGB 图像
        mask: 形状为 (H, W) 或 (1, H, W) 的二值掩码
        color: RGB 颜色；为 None 时使用白色
        thickness: 轮廓线宽

    返回：
        形状为 (H, W, 3) 的 uint8 图像，带有轮廓
    """
    if color is None:
        color = [255, 255, 255]

    mask_2d = mask.squeeze().astype(np.uint8)
    # OpenCV 使用 BGR 格式
    result_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    bgr_color = (int(color[2]), int(color[1]), int(color[0]))
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contours, -1, bgr_color, thickness=thickness)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def apply_colormap(
    feature_map: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    对单通道特征图应用伪彩色映射。

    参数：
        feature_map: 形状为 (H, W) 的浮点数组
        colormap: OpenCV 颜色映射常量，默认为 COLORMAP_JET

    返回：
        形状为 (H, W, 3) 的 uint8 RGB 图像
    """
    normalized = feature_map - feature_map.min()
    denom = feature_map.max() - feature_map.min()
    if denom > 1e-8:
        normalized = normalized / denom
    gray = (normalized * 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(gray, colormap)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
