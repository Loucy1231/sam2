#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SAM2 视频分割可视化演示脚本

对视频文件或图像目录逐帧运行 SAM2 推理，并输出带有分割掩码叠加的视频。

用法示例：
    # 处理视频文件（使用点提示初始化第 0 帧）
    python demo_video_segmentation.py --video notebooks/videos/bedroom.mp4 \
        --point 300 200 --label 1 --output output/

    # 处理图像目录
    python demo_video_segmentation.py --frames-dir /path/to/frames/ \
        --point 300 200 --label 1 --output output/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualizations import VideoVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 视频分割可视化演示")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="输入视频文件路径（.mp4、.avi 等）")
    src.add_argument("--frames-dir", type=str, help="输入帧图像目录路径")

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
        help="第 0 帧的提示点坐标 (x y)，可多次使用",
    )
    parser.add_argument(
        "--label",
        type=int,
        action="append",
        default=None,
        help="提示点标签（1=前景，0=背景），与 --point 一一对应",
    )
    parser.add_argument("--output", type=str, default="output/", help="输出目录")
    parser.add_argument("--fps", type=float, default=None, help="输出视频帧率（默认与原视频相同）")
    parser.add_argument("--alpha", type=float, default=0.5, help="掩码透明度 [0,1]")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数（用于快速测试）")
    return parser.parse_args()


def extract_frames_from_video(video_path: str, max_frames=None):
    """从视频文件提取 RGB 帧列表，返回 (frames, fps)。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, orig_fps


def load_frames_from_dir(frames_dir: str, max_frames=None):
    """从目录加载 PNG/JPG 帧，返回 RGB 帧列表。"""
    extensions = (".jpg", ".jpeg", ".png")
    paths = sorted(
        p for p in Path(frames_dir).iterdir() if p.suffix.lower() in extensions
    )
    if max_frames:
        paths = paths[:max_frames]
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


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
    # 检查模型权重
    # ------------------------------------------------------------------
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"[错误] 找不到权重文件：{checkpoint}")
        print("请先下载模型权重，参考 README 中的说明。")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 加载视频帧
    # ------------------------------------------------------------------
    orig_fps = args.fps or 30.0

    if args.video:
        video_path = args.video
        if not Path(video_path).exists():
            print(f"[错误] 找不到视频文件：{video_path}")
            sys.exit(1)
        print(f"加载视频：{video_path}")
        frames, orig_fps = extract_frames_from_video(video_path, max_frames=args.max_frames)
    else:
        frames_dir = args.frames_dir
        if not Path(frames_dir).exists():
            print(f"[错误] 找不到帧目录：{frames_dir}")
            sys.exit(1)
        print(f"加载帧目录：{frames_dir}")
        frames = load_frames_from_dir(frames_dir, max_frames=args.max_frames)

    if len(frames) == 0:
        print("[错误] 未加载到任何帧。")
        sys.exit(1)

    fps = args.fps or orig_fps
    print(f"共加载 {len(frames)} 帧，帧率 {fps:.1f} fps")

    # ------------------------------------------------------------------
    # 准备提示点
    # ------------------------------------------------------------------
    if args.point:
        point_coords = np.array(args.point)
        labels = args.label if args.label else [1] * len(args.point)
        point_labels = np.array(labels)
    else:
        h, w = frames[0].shape[:2]
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])
        print(f"未提供提示，使用第 0 帧中心点：{point_coords[0]}")

    # ------------------------------------------------------------------
    # 保存临时帧目录供 SAM2VideoPredictor 使用
    # ------------------------------------------------------------------
    import tempfile

    tmp_frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(f"{tmp_frames_dir}/{i:05d}.jpg")

    # ------------------------------------------------------------------
    # 加载模型并推理
    # ------------------------------------------------------------------
    from sam2.build_sam import build_sam2_video_predictor

    print(f"加载模型：{checkpoint}")
    predictor = build_sam2_video_predictor(args.model_cfg, str(checkpoint), device=device)

    print("初始化推理状态……")
    with torch.inference_mode():
        inference_state = predictor.init_state(video_path=tmp_frames_dir)
        predictor.reset_state(inference_state)

        # 在第 0 帧添加点提示
        obj_id = 1
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=point_coords,
            labels=point_labels,
        )

        # 逐帧传播
        print("传播分割结果到所有帧……")
        masks_per_frame = []
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            frame_masks = {}
            for oid, logit in zip(obj_ids, mask_logits):
                binary_mask = (logit[0] > 0.0).cpu().numpy()
                frame_masks[oid] = binary_mask
            masks_per_frame.append(frame_masks)

    # ------------------------------------------------------------------
    # 可视化并写出视频
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "segmentation_result.mp4"

    viz = VideoVisualizer(alpha=args.alpha, fps=fps)
    print("渲染可视化视频……")
    viz.create_segmentation_video(frames, masks_per_frame, output_video, fps=fps)
    print(f"\n可视化视频已保存：{output_video.resolve()}")

    # 预览第一帧
    if masks_per_frame:
        preview_path = output_dir / "preview_frame0.png"
        viz.show_frame(frames[0], masks_per_frame[0], frame_idx=0, save_path=preview_path)
        print(f"第 0 帧预览已保存：{preview_path.resolve()}")

    # 清理临时目录
    import shutil
    shutil.rmtree(tmp_frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
