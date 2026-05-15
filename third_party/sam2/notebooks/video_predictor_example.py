#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright (c) Meta Platforms, Inc. and affiliates.


# # Video segmentation with SAM 2

# This notebook shows how to use SAM 2 for interactive segmentation in videos. It will cover the following:
# 
# - adding clicks (or box) on a frame to get and refine _masklets_ (spatio-temporal masks)
# - propagating clicks (or box) to get _masklets_ throughout the video
# - segmenting and tracking multiple objects at the same time
# 
# We use the terms _segment_ or _mask_ to refer to the model prediction for an object on a single frame, and _masklet_ to refer to the spatio-temporal masks across the entire video. 

# <a target="_blank" href="https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb">
#   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
# </a>

# ## Environment Set-up

# If running locally using jupyter, first install `sam2` in your environment using the [installation instructions](https://github.com/facebookresearch/sam2#installation) in the repository.
# 
# If running from Google Colab, set `using_colab=True` below and run the cell. In Colab, be sure to select 'GPU' under 'Edit'->'Notebook Settings'->'Hardware accelerator'. Note that it's recommended to use **A100 or L4 GPUs when running in Colab** (T4 GPUs might also work, but could be slow and might run out of memory in some cases).

# In[2]:


using_colab = False


# In[3]:


if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())
    import sys
    get_ipython().system('{sys.executable} -m pip install opencv-python matplotlib')
    get_ipython().system("{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam2.git'")

    get_ipython().system('mkdir -p videos')
    get_ipython().system('wget -P videos https://dl.fbaipublicfiles.com/segment_anything_2/assets/bedroom.zip')
    get_ipython().system('unzip -d videos videos/bedroom.zip')

    get_ipython().system('mkdir -p ../checkpoints/')
    get_ipython().system('wget -P ../checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt')


# ## Set-up

# In[4]:


import os
import sys
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import argparse
from pathlib import Path

_parser = argparse.ArgumentParser(description="SAM2 video predictor example")
_parser.add_argument(
    "--video-dir",
    required=True,
    help="Directory of video frames (jpg/png, sortable numeric stem in filename).",
)
_parser.add_argument(
    "--mask-out-dir",
    required=True,
    help="Directory to write per-frame mask PNGs (000000.png, ...).",
)
_parser.add_argument(
    "--ann-frame",
    type=int,
    default=0,
    help="Frame index for single-stage prompts (clamped to valid range).",
)
_parser.add_argument(
    "--prompt",
    choices=("points", "box", "box_gui"),
    default="points",
    help=(
        "Single-stage prompt: points (stdin multi-line x y [label]); "
        "box (stdin one line x_min y_min x_max y_max); "
        "box_gui (mouse drag rectangle on image, needs DISPLAY / GUI)."
    ),
)
_parser.add_argument(
    "--run-extra-examples",
    action="store_true",
    help="After single-stage, run original notebook demos (more stdin + propagations).",
)
_parser.add_argument(
    "--num-objects",
    type=int,
    default=1,
    help=(
        "同一 conditioning 帧上要跟踪的物体数量（>=1）。"
        "依次采集每个物体的点或框（obj_id 为 1..N），共用一次 propagate；"
        "写出 mask 时默认合并所有物体。"
    ),
)
_cli_args = _parser.parse_args()
if _cli_args.num_objects < 1:
    raise SystemExit("--num-objects 须为 >= 1 的整数")
_num_objects = _cli_args.num_objects
video_dir = os.path.abspath(os.path.expanduser(_cli_args.video_dir))
mask_out_dir = Path(os.path.abspath(os.path.expanduser(_cli_args.mask_out_dir)))
if not os.path.isdir(video_dir):
    raise SystemExit(f"--video-dir is not a directory: {video_dir}")
mask_out_dir.mkdir(parents=True, exist_ok=True)

if _cli_args.prompt == "box_gui":
    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise SystemExit(
                "--prompt box_gui 需要图形界面（设置 DISPLAY 或 WAYLAND_DISPLAY，"
                "或使用 SSH -X/-Y）。无界面时请用 --prompt box 在终端输入四角坐标。"
            )
    import matplotlib

    for _backend in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue
    else:
        print(
            "[WARN] 未能设置交互式 Matplotlib 后端（TkAgg/Qt）；"
            "若窗口无法打开，请安装 python3-tk 或 PyQt5。"
        )

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

if _cli_args.prompt == "box_gui":
    _bk = matplotlib.get_backend().lower()
    if "agg" in _bk:
        raise SystemExit(
            "Matplotlib 后端为 Agg，无法弹出交互窗口。请在 sam2 环境中安装 GUI 支持并启用交互后端，例如:\n"
            "  conda install tk\n"
            "  或: pip install pyqt5\n"
            "  然后: export MPLBACKEND=TkAgg\n"
            "远程无桌面时请改用: --prompt box"
        )
    print(
        "[box_gui] bash 多行命令时，除最后一行外每行行尾都要有反斜杠 \\ ，"
        "否则 '--prompt box_gui' 不会传给 Python（脚本会退回默认的点提示）。\n"
    )


# In[5]:


# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


# ### Loading the SAM 2 video predictor

# In[6]:


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)


# In[7]:


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def print_frame_pixel_info(frame_path):
    """在终端打印该帧的宽高（像素），便于输入 x, y 时对齐范围。"""
    with Image.open(frame_path) as im:
        w, h = im.size
    print(f"\n[帧] {frame_path}")
    print(f"  宽高（像素）: width={w}, height={h}")
    print(
        f"  坐标系: 原点左上，x 向右 0..{w - 1}，y 向下 0..{h - 1}\n"
    )
    return w, h


def read_points_labels_interactive(
    frame_path, existing_points=None, existing_labels=None, allow_empty=False
):
    """
    先 print_frame_pixel_info，再从 stdin 读入多点。
    每行: `x y` 或 `x y label`（label 可选，1=前景，0=背景，默认 1）。
    空行结束。若提供 existing_*，会先列出已有点再追加新行。
    若 allow_empty 为 True 且未输入任何点（也无 existing），返回 (None, None) 而非退出。
    """
    print_frame_pixel_info(frame_path)
    ep = (
        np.zeros((0, 2), dtype=np.float32)
        if existing_points is None or len(existing_points) == 0
        else np.asarray(existing_points, dtype=np.float32)
    )
    el = (
        np.zeros((0,), dtype=np.int32)
        if existing_labels is None or len(existing_labels) == 0
        else np.asarray(existing_labels, dtype=np.int32)
    )
    new_pts, new_lbl = [], []
    if len(ep) > 0:
        print("已输入的点（将保留并与下面追加的点合并后传给 SAM2）:")
        for i in range(len(ep)):
            print(f"  ({ep[i, 0]:.0f}, {ep[i, 1]:.0f})  label={int(el[i])}")
        print("请继续输入要追加的点（每行 x y [label]），仅按回车空行结束追加。\n")
    else:
        if allow_empty:
            print(
                "可选：输入修正点（每行 x y [label]，1=前景 0=背景），"
                "仅按回车空行则跳过。\n"
            )
        else:
            print("请输入点提示（每行 x y [label]），空行结束。\n")

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            print("  忽略：至少需要 x 与 y 两个数")
            continue
        x, y = float(parts[0]), float(parts[1])
        lab = int(float(parts[2])) if len(parts) >= 3 else 1
        if lab not in (0, 1):
            print("  label 须为 0 或 1，已按 1 处理")
            lab = 1
        new_pts.append((x, y))
        new_lbl.append(lab)

    if new_pts:
        ep = np.vstack([ep, np.asarray(new_pts, dtype=np.float32)])
        el = np.concatenate([el, np.asarray(new_lbl, dtype=np.int32)])
    if len(ep) == 0:
        if allow_empty:
            return None, None
        raise SystemExit("未输入任何点；请至少输入一行 x y（或带 label）。")
    return ep, el


def read_box_interactive(frame_path):
    """先打印图像尺寸，再读入一行 x_min y_min x_max y_max（像素）。"""
    print_frame_pixel_info(frame_path)
    print("请输入矩形框: x_min y_min x_max y_max（一行，空格分隔）\n")
    line = input("> ").strip()
    parts = line.replace(",", " ").split()
    if len(parts) != 4:
        raise SystemExit("框需要四个数: x_min y_min x_max y_max")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)


def read_box_matplotlib_gui(frame_path):
    """
    在弹出窗口中用鼠标拖出轴对齐矩形，关闭窗口后提交。
    返回与 read_box_interactive 相同格式的 float32 数组 [x_min, y_min, x_max, y_max]（像素）。
    """
    from matplotlib.widgets import RectangleSelector

    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise SystemExit(
                "box_gui 需要图形界面。无 DISPLAY 时请改用 --prompt box。"
            )

    with Image.open(frame_path) as im:
        w, h = im.size
        arr = np.asarray(im)
    print_frame_pixel_info(frame_path)
    print(
        "将弹出独立窗口：请用鼠标左键拖动绘制矩形（可拖拽调整）；"
        "完成后关闭窗口即可继续（无需在终端按回车）。\n"
    )

    state = {"extents": None}

    def onselect(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        ymin = min(eclick.ydata, erelease.ydata)
        ymax = max(eclick.ydata, erelease.ydata)
        state["extents"] = (xmin, xmax, ymin, ymax)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(arr, origin="upper")
    ax.set_title("Drag rectangle, then close this window")

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords="data",
        interactive=True,
    )

    def snapshot_selector_extents():
        ext = getattr(selector, "extents", None)
        if ext is None:
            return
        try:
            bx0, bx1, by0, by1 = float(ext[0]), float(ext[1]), float(ext[2]), float(ext[3])
            if bx1 > bx0 and by1 > by0:
                state["extents"] = (bx0, bx1, by0, by1)
        except (TypeError, ValueError, IndexError):
            pass

    fig.canvas.mpl_connect(
        "button_release_event",
        lambda _ev: snapshot_selector_extents(),
    )

    def on_close(_evt):
        snapshot_selector_extents()

    fig.canvas.mpl_connect("close_event", on_close)

    plt.tight_layout()
    fig.canvas.draw()

    try:
        mgr = fig.canvas.manager
        if mgr is not None:
            win = getattr(mgr, "window", None)
            if win is not None:
                if hasattr(win, "lift"):
                    win.lift()
                if hasattr(win, "activateWindow"):
                    win.activateWindow()
                if hasattr(win, "wm_attributes"):
                    try:
                        win.wm_attributes("-topmost", True)
                        win.after(200, lambda: win.wm_attributes("-topmost", False))
                    except Exception:
                        pass
    except Exception:
        pass

    plt.show(block=True)

    if state["extents"] is None:
        raise SystemExit(
            "未选择有效矩形框：请在窗口中拖动拉出矩形后再关闭窗口。"
        )
    xmin, xmax, ymin, ymax = state["extents"]

    xmin = float(np.clip(xmin, 0, w))
    xmax = float(np.clip(xmax, 0, w))
    ymin = float(np.clip(ymin, 0, h))
    ymax = float(np.clip(ymax, 0, h))
    if xmax <= xmin or ymax <= ymin:
        raise SystemExit("矩形框无效（宽与高须为正）。")
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)


# #### Select an example video

# We assume that the video is stored as a list of JPEG frames with filenames like `<frame_index>.jpg`.
# 
# For your custom videos, you can extract their JPEG frames using ffmpeg (https://ffmpeg.org/) as follows:
# ```
# ffmpeg -i <your_video>.mp4 -q:v 2 -start_number 0 <output_dir>/'%05d.jpg'
# ```
# where `-q:v` generates high-quality JPEG frames and `-start_number 0` asks ffmpeg to start the JPEG file from `00000.jpg`.

# In[ ]:


# `video_dir` / `mask_out_dir` are set from CLI: --video-dir, --mask-out-dir

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


# #### Initialize the inference state

# SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an **inference state** on this video.
# 
# During initialization, it loads all the JPEG frames in `video_path` and stores their pixels in `inference_state` (as shown in the progress bar below).

# In[9]:


inference_state = predictor.init_state(video_path=video_dir)


# ### Single-stage: 同一帧上一次性输入全部点 -> 全视频传播

predictor.reset_state(inference_state)

ann_frame_idx = max(0, min(_cli_args.ann_frame, len(frame_names) - 1))
_conditioning_frame_idx = ann_frame_idx

_multi_hint = (
    f"  当前将添加 {_num_objects} 个物体（--num-objects）；每个物体使用 obj_id=1..{_num_objects}。\n"
    if _num_objects > 1
    else ""
)

if _cli_args.prompt == "points":
    print(
        "\n[单阶段 · 点提示] 在 conditioning 帧上为每个物体分别输入前/背景点（每物体：多行 `x y [label]`，"
        "空行结束该物体）。\n"
        f"  当前 conditioning 帧索引: {ann_frame_idx}（可用 --ann-frame 修改）。\n"
        + _multi_hint
        + "  改用矩形框请传: --prompt box 或 --prompt box_gui。\n"
        "  若还要跑原版 notebook 里后续多段演示，请加参数: --run-extra-examples\n"
    )
elif _cli_args.prompt == "box":
    print(
        "\n[单阶段 · 框提示 · 终端] 每个物体：一行矩形框 x_min y_min x_max y_max（像素）；"
        "可选再输入该物体的前/背景修正点（空行跳过）。\n"
        f"  当前 conditioning 帧索引: {ann_frame_idx}（可用 --ann-frame 修改）。\n"
        + _multi_hint
        + "  鼠标拖框请用: --prompt box_gui；多点请用: --prompt points\n"
        "  若还要跑原版 notebook 里后续多段演示，请加参数: --run-extra-examples\n"
    )
else:
    print(
        "\n[单阶段 · 框提示 · 鼠标] 每个物体在弹出窗口中拖一个矩形；"
        "可选再输入该物体的前/背景修正点（空行跳过）。\n"
        f"  当前 conditioning 帧索引: {ann_frame_idx}（可用 --ann-frame 修改）。\n"
        + _multi_hint
        + "  无图形界面请用: --prompt box；多点请用: --prompt points\n"
        "  若还要跑原版 notebook 里后续多段演示，请加参数: --run-extra-examples\n"
    )

_interact_path = os.path.join(video_dir, frame_names[ann_frame_idx])
# 多物体：框模式记录每个 obj 的框与可选修正点；点模式记录每物体点（用于可视化）
boxes_by_obj = {}
points_prompt_by_obj = {}
points_refine_by_obj = {}
out_obj_ids, out_mask_logits = None, None

for obj_slot in range(_num_objects):
    ann_obj_id = obj_slot + 1
    if _num_objects > 1:
        print(f"\n========== 物体 {ann_obj_id}/{_num_objects} (obj_id={ann_obj_id}) ==========\n")

    if _cli_args.prompt == "points":
        points, labels = read_points_labels_interactive(_interact_path)
        points_prompt_by_obj[ann_obj_id] = (points, labels)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
    else:
        if _cli_args.prompt == "box":
            box = read_box_interactive(_interact_path)
        else:
            box = read_box_matplotlib_gui(_interact_path)
        boxes_by_obj[ann_obj_id] = box
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )
        print(
            f"\n已提交物体 {ann_obj_id} 的框（像素）: {box.tolist()}；"
            "如需用前/背景点修正该物体当前帧 mask，见下方提示；不需要则直接空行跳过。\n"
        )
        points_ref, labels_ref = read_points_labels_interactive(
            _interact_path, allow_empty=True
        )
        if points_ref is not None and len(points_ref) > 0:
            points, labels = points_ref, labels_ref
            points_refine_by_obj[ann_obj_id] = (points, labels)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
                box=box,
            )
        else:
            points_refine_by_obj[ann_obj_id] = None
            points = np.zeros((0, 2), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int32)

if _cli_args.prompt == "points":
    points, labels = points_prompt_by_obj[_num_objects]
    box = None
else:
    box = boxes_by_obj.get(_num_objects)

plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx} ({_num_objects} object(s))")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
if _cli_args.prompt in ("box", "box_gui"):
    for oid in sorted(boxes_by_obj):
        show_box(boxes_by_obj[oid], plt.gca())
    for oid in sorted(points_refine_by_obj):
        pl = points_refine_by_obj[oid]
        if pl is not None and len(pl[1]) > 0:
            show_points(pl[0], pl[1], plt.gca())
else:
    for oid in sorted(points_prompt_by_obj):
        pt, lb = points_prompt_by_obj[oid]
        show_points(pt, lb, plt.gca())
assert out_obj_ids is not None and out_mask_logits is not None
for i, oid in enumerate(out_obj_ids):
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=oid)

vis_frame_stride = 30

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

if _cli_args.run_extra_examples:
    print("\n[INFO] --run-extra-examples：以下为原版 notebook 后续段落（多轮 stdin / reset / 传播）。\n")
    if _num_objects > 1:
        print(
            "[WARN] 单阶段已用 --num-objects > 1；下面 extra 段主要针对单物体示例，"
            "追加点时的 existing 仅含上一阶段最后一个物体的点，如需精调请自行改脚本。\n"
        )

    # #### Step 2 (notebook): 在同 conditioning 帧上再追加一轮点
    ann_frame_idx = _conditioning_frame_idx
    ann_obj_id = 1
    _interact_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    points, labels = read_points_labels_interactive(
        _interact_path, existing_points=points, existing_labels=labels
    )
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    # Step 4–5: 另一帧上 refine
    ann_frame_idx = min(150, len(frame_names) - 1)
    ann_obj_id = 1
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx} -- before refinement")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_mask(video_segments[ann_frame_idx][ann_obj_id], plt.gca(), obj_id=ann_obj_id)
    _refine_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    points, labels = read_points_labels_interactive(_refine_path)
    _, _, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx} -- after refinement")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits > 0.0).cpu().numpy(), plt.gca(), obj_id=ann_obj_id)

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    # Example 2: box
    predictor.reset_state(inference_state)
    ann_frame_idx = 0
    ann_obj_id = 4
    _box_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    box = read_box_interactive(_box_path)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_box(box, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    ann_frame_idx = 0
    ann_obj_id = 4
    _ann = os.path.join(video_dir, frame_names[ann_frame_idx])
    print(f"\n沿用框（像素）: {box.tolist()}；下面输入 refinement 点。")
    points, labels = read_points_labels_interactive(_ann)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_box(box, plt.gca())
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    # Example 3: two objects
    predictor.reset_state(inference_state)
    prompts = {}
    ann_frame_idx = 0
    ann_obj_id = 2
    _mpath = os.path.join(video_dir, frame_names[ann_frame_idx])
    points, labels = read_points_labels_interactive(_mpath)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    ann_frame_idx = 0
    ann_obj_id = 2
    _mpath = os.path.join(video_dir, frame_names[ann_frame_idx])
    points, labels = read_points_labels_interactive(
        _mpath, existing_points=points, existing_labels=labels
    )
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    ann_frame_idx = 0
    ann_obj_id = 3
    _mpath = os.path.join(video_dir, frame_names[ann_frame_idx])
    points, labels = read_points_labels_interactive(_mpath)
    prompts[ann_obj_id] = points, labels
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    plt.figure(figsize=(9, 6))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

# In[ ]:

# None: 合并所有目标；如果只导出单目标，改成对应 obj_id（如 1）
target_obj_id = None

for frame_idx in range(len(frame_names)):
    merged_mask = None

    if frame_idx in video_segments:
        if target_obj_id is None:
            for _, out_mask in video_segments[frame_idx].items():
                m = np.squeeze(out_mask).astype(bool)
                merged_mask = m if merged_mask is None else (merged_mask | m)
        elif target_obj_id in video_segments[frame_idx]:
            merged_mask = np.squeeze(video_segments[frame_idx][target_obj_id]).astype(bool)

    if merged_mask is None:
        h, w = np.array(Image.open(Path(video_dir) / frame_names[frame_idx])).shape[:2]
        mask_u8 = np.zeros((h, w), dtype=np.uint8)
    else:
        mask_u8 = merged_mask.astype(np.uint8) * 255

    Image.fromarray(mask_u8, mode="L").save(mask_out_dir / f"{frame_idx:06d}.png")

print(f"[INFO] Saved masks to: {mask_out_dir}")
if _cli_args.run_extra_examples:
    print(
        "[INFO] 已使用 --run-extra-examples：写出的是脚本中最后一次 propagate 的结果（通常为末尾多物体示例）。"
    )

