# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
视频分割可视化模块

提供基于 SAM2 的视频分割结果可视化功能，支持：
- 逐帧处理并叠加分割掩码
- 输出带有分割结果的视频文件
- 多目标彩色编码显示
- 支持多种视频编解码器
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualizations.utils import create_color_palette, draw_boundaries, overlay_mask_on_image


class VideoVisualizer:
    """
    视频分割结果可视化器。

    使用示例：
        from visualizations import VideoVisualizer

        viz = VideoVisualizer(alpha=0.5)
        viz.create_segmentation_video(
            frames, masks_per_frame, output_path="output.mp4"
        )
    """

    def __init__(
        self,
        alpha: float = 0.5,
        show_borders: bool = True,
        fps: float = 30.0,
        codec: str = "mp4v",
    ) -> None:
        """
        初始化视频可视化器。

        参数：
            alpha: 掩码叠加透明度，范围 [0, 1]
            show_borders: 是否绘制分割边界
            fps: 输出视频帧率
            codec: 视频编解码器四字符代码，如 'mp4v'、'avc1'
        """
        self.alpha = alpha
        self.show_borders = show_borders
        self.fps = fps
        self.codec = codec

    # ------------------------------------------------------------------
    # 帧级可视化
    # ------------------------------------------------------------------

    def render_frame(
        self,
        frame: np.ndarray,
        masks: Union[np.ndarray, Dict[int, np.ndarray]],
        palette: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        将分割掩码渲染到单帧图像上。

        参数：
            frame: 形状为 (H, W, 3) 的 uint8 RGB 图像
            masks: 形状为 (N, H, W) 的掩码数组，或以目标 ID 为键的掩码字典
            palette: 颜色调色板；为 None 时自动生成

        返回：
            形状为 (H, W, 3) 的 uint8 RGB 渲染图像
        """
        if isinstance(masks, dict):
            mask_list = list(masks.values())
        else:
            mask_list = [masks[i] for i in range(len(masks))]

        if palette is None:
            palette = create_color_palette(len(mask_list))

        result = frame.copy()
        for i, mask in enumerate(mask_list):
            if mask is None:
                continue
            mask_2d = mask.squeeze()
            if not np.any(mask_2d):
                continue
            color = palette[i % len(palette)].tolist()
            result = overlay_mask_on_image(result, mask_2d, color=color, alpha=self.alpha)
            if self.show_borders:
                result = draw_boundaries(result, mask_2d, color=color, thickness=2)

        return result

    # ------------------------------------------------------------------
    # 视频写入
    # ------------------------------------------------------------------

    def create_segmentation_video(
        self,
        frames: Union[List[np.ndarray], np.ndarray],
        masks_per_frame: Union[
            List[np.ndarray],           # 每帧对应形状 (N, H, W) 的掩码数组
            List[Dict[int, np.ndarray]],  # 每帧对应以目标 ID 为键的掩码字典
        ],
        output_path: Union[str, Path],
        fps: Optional[float] = None,
    ) -> Path:
        """
        将分割掩码渲染到每帧并写出视频文件。

        参数：
            frames: RGB 帧列表（每帧 uint8 (H, W, 3)）或形状为 (T, H, W, 3) 的数组
            masks_per_frame: 与 frames 等长的掩码序列
            output_path: 输出视频文件路径（推荐 .mp4）
            fps: 帧率；为 None 时使用初始化参数

        返回：
            保存后的文件路径
        """
        if fps is None:
            fps = self.fps

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frames_list = list(frames)
        n_frames = len(frames_list)
        if n_frames == 0:
            raise ValueError("frames 不能为空。")

        h, w = frames_list[0].shape[:2]

        # 自动检测最大目标数以确定调色板大小
        n_objects = 1
        for m in masks_per_frame:
            if isinstance(m, dict):
                n_objects = max(n_objects, len(m))
            elif hasattr(m, "__len__"):
                n_objects = max(n_objects, len(m))
        palette = create_color_palette(n_objects)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        # OpenCV VideoWriter 使用 BGR 格式
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame, masks in zip(frames_list, masks_per_frame):
            rendered_rgb = self.render_frame(frame, masks, palette=palette)
            rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
            writer.write(rendered_bgr)

        writer.release()
        return output_path

    def create_video_from_dir(
        self,
        frames_dir: Union[str, Path],
        masks_per_frame: Union[
            List[np.ndarray],
            List[Dict[int, np.ndarray]],
        ],
        output_path: Union[str, Path],
        fps: Optional[float] = None,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    ) -> Path:
        """
        从图像目录读取帧，叠加掩码后写出视频。

        参数：
            frames_dir: 包含帧图像的目录路径
            masks_per_frame: 与帧图像等长的掩码序列
            output_path: 输出视频文件路径
            fps: 帧率（可选）
            extensions: 支持的图像扩展名

        返回：
            保存后的文件路径
        """
        frames_dir = Path(frames_dir)
        frame_paths = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in extensions
        )
        if len(frame_paths) == 0:
            raise ValueError(f"目录 {frames_dir} 中未找到图像文件。")

        frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths]
        return self.create_segmentation_video(frames, masks_per_frame, output_path, fps=fps)

    # ------------------------------------------------------------------
    # 单帧预览
    # ------------------------------------------------------------------

    def show_frame(
        self,
        frame: np.ndarray,
        masks: Union[np.ndarray, Dict[int, np.ndarray]],
        frame_idx: int = 0,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """
        显示单帧的分割结果。

        参数：
            frame: 形状为 (H, W, 3) 的 uint8 RGB 帧
            masks: 该帧的掩码（数组或字典）
            frame_idx: 帧索引（仅用于标题显示）
            figsize: 图形尺寸
            save_path: 保存路径（可选）

        返回：
            matplotlib Figure 对象
        """
        rendered = self.render_frame(frame, masks)

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
        axes[0].imshow(frame)
        axes[0].set_title(f"原始帧（第 {frame_idx} 帧）", fontsize=13)
        axes[0].axis("off")

        axes[1].imshow(rendered)
        axes[1].set_title(f"分割结果（第 {frame_idx} 帧）", fontsize=13)
        axes[1].axis("off")

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        return fig
