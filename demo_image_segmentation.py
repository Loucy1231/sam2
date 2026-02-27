#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 单张图片分割可视化演示脚本

用法示例：
    # 使用点提示
    python demo_image_segmentation.py --image notebooks/images/truck.jpg \
        --point 500 375 --label 1

    # 使用边界框提示
    python demo_image_segmentation.py --image notebooks/images/truck.jpg \
        --box 425 600 700 875

    # 保存可视化结果
    python demo_image_segmentation.py --image notebooks/images/truck.jpg \
        --point 500 375 --label 1 --output output_dir/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualizations import ImageVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 图像分割可视化演示")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 模型权重路径",
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 模型配置文件路径",
    )
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        action="append",
        default=None,
        help="提示点坐标 (x y)，可多次使用",
    )
    parser.add_argument(
        "--label",
        type=int,
        action="append",
        default=None,
        help="提示点标签（1=前景，0=背景），与 --point 一一对应",
    )
    parser.add_argument(
        "--box",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="边界框提示 (x0 y0 x1 y1)",
    )
    parser.add_argument("--output", type=str, default=None, help="可视化结果输出目录")
    parser.add_argument("--alpha", type=float, default=0.5, help="掩码透明度 [0,1]")
    parser.add_argument("--no-borders", action="store_true", help="不显示分割边界")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 设备配置
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")

    # ------------------------------------------------------------------
    # 加载模型
    # ------------------------------------------------------------------
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = Path(args.checkpoint)
    model_cfg = args.model_cfg

    if not checkpoint.exists():
        print(f"[错误] 找不到权重文件：{checkpoint}")
        print("请先下载模型权重，参考 README 中的说明。")
        sys.exit(1)

    print(f"加载模型：{checkpoint}")
    sam2_model = build_sam2(model_cfg, str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # ------------------------------------------------------------------
    # 加载图像
    # ------------------------------------------------------------------
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[错误] 找不到图像文件：{image_path}")
        sys.exit(1)

    image = np.array(Image.open(image_path).convert("RGB"))
    print(f"图像尺寸：{image.shape}")

    # ------------------------------------------------------------------
    # 准备提示
    # ------------------------------------------------------------------
    point_coords = None
    point_labels = None
    box = None

    if args.point:
        point_coords = np.array(args.point)
        labels = args.label if args.label else [1] * len(args.point)
        point_labels = np.array(labels)

    if args.box:
        box = np.array(args.box)

    if point_coords is None and box is None:
        # 使用默认中心点作为演示
        h, w = image.shape[:2]
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        print(f"未提供提示，使用图像中心点：{point_coords[0]}")

    # ------------------------------------------------------------------
    # 运行推理
    # ------------------------------------------------------------------
    print("运行 SAM2 推理……")
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(image)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True,
                )
        else:
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

    print(f"生成掩码数量：{len(masks)}，置信度：{scores}")

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------
    viz = ImageVisualizer(alpha=args.alpha, show_borders=not args.no_borders)

    output_dir = Path(args.output) if args.output else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 显示所有掩码
    fig_all = viz.show_segmentation(
        image, masks, scores,
        point_coords=point_coords, point_labels=point_labels, box=box,
        title="SAM2 图像分割结果",
        save_path=output_dir / "all_masks.png",
    )

    # 显示最佳掩码
    fig_best = viz.show_best_mask(
        image, masks, scores,
        point_coords=point_coords, point_labels=point_labels, box=box,
        title="最佳分割掩码",
        save_path=output_dir / "best_mask.png",
    )

    # 并排对比
    fig_cmp = viz.show_comparison(
        image, masks, scores,
        point_coords=point_coords, point_labels=point_labels, box=box,
        title="分割结果对比",
        save_path=output_dir / "comparison.png",
    )

    # 保存叠加图
    viz.save_overlay(image, masks, output_dir / "overlay.png", scores=scores)

    print(f"\n可视化结果已保存到：{output_dir.resolve()}")
    print("  - all_masks.png   : 所有候选掩码")
    print("  - best_mask.png   : 最佳掩码")
    print("  - comparison.png  : 三视图对比")
    print("  - overlay.png     : 彩色叠加图")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
