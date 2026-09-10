"""Render video camminata a terra da una run salvata.

Uso:
    python3 viz_scripts/walkthrough.py configs/found/dgsg_dynamic.py \
        --out experiments/FOUND/00824_dynamic_0/walk.mp4 --fps 10 --frames 200

Segue la traiettoria stimata smussata e livellata (utils/walk_viz):
quota costante a altezza occhi, sguardo orizzontale, niente salti.
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation
from utils.walk_viz import smooth_walk_trajectory, resample_trajectory


def load_all(scene_path):
    d = dict(np.load(scene_path, allow_pickle=True))
    w2c0 = d["w2c"]
    intr = d["intrinsics"]
    k = np.array(intr[:3, :3], dtype=np.float64)
    n = d["cam_unnorm_rots"].shape[-1]
    all_w2cs = []
    for t in range(n):
        rot = F.normalize(torch.tensor(d["cam_unnorm_rots"][..., t]).float())
        tran = torch.tensor(d["cam_trans"][..., t]).float()
        rel = torch.eye(4)
        rel[:3, :3] = build_rotation(rot)
        rel[:3, 3] = tran
        all_w2cs.append(rel.numpy())
    params = {kk: torch.tensor(d[kk]).cuda().float() for kk in
              ["means3D", "rgb_colors", "unnorm_rotations", "logit_opacities", "log_scales"]}
    return params, w2c0, k, all_w2cs, d


def render_frame(params, w2c, k, W, H):
    cam = setup_camera(W, H, k, w2c, 0.01, 100.0)
    white = Camera(image_height=cam.image_height, image_width=cam.image_width,
                   tanfovx=cam.tanfovx, tanfovy=cam.tanfovy,
                   bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
                   scale_modifier=cam.scale_modifier, viewmatrix=cam.viewmatrix,
                   projmatrix=cam.projmatrix, sh_degree=cam.sh_degree,
                   campos=cam.campos, prefiltered=cam.prefiltered)
    log_scales = params["log_scales"]
    if log_scales.shape[-1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    rendervar = {"means3D": params["means3D"], "colors_precomp": params["rgb_colors"],
                 "rotations": F.normalize(params["unnorm_rotations"]),
                 "opacities": torch.sigmoid(params["logit_opacities"]),
                 "scales": torch.exp(log_scales),
                 "means2D": torch.zeros_like(params["means3D"])}
    with torch.no_grad():
        im, _, _ = Renderer(raster_settings=white)(**rendervar)
    im = torch.clamp(im, 0, 1).permute(1, 2, 0).cpu().numpy()
    return (im * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--eye", type=float, default=None)
    ap.add_argument("--smooth", type=int, default=15)
    ap.add_argument("--no-level", action="store_true")
    args = ap.parse_args()

    exp = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    seed_everything(seed=exp.config["seed"])
    viz = exp.config["viz"]
    results_dir = os.path.join(exp.config["workdir"], exp.config["run_name"])
    scene_path = exp.config.get("scene_path", os.path.join(results_dir, "params_with_idx.npz"))

    params, _, k, all_w2cs, _ = load_all(scene_path)
    W, H = viz["viz_w"], viz["viz_h"]
    k = k.copy()
    k[0, :] *= W / 640
    k[1, :] *= H / 480

    walk = smooth_walk_trajectory(all_w2cs, smooth_window=args.smooth,
                                  eye_height=args.eye, level_camera=not args.no_level)
    walk = resample_trajectory(walk, args.frames)

    out = args.out or os.path.join(results_dir, "walk.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for w2c in tqdm(walk, desc="walkthrough"):
        frame = render_frame(params, np.array(w2c), k, W, H)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"scritto {out}")


if __name__ == "__main__":
    main()
