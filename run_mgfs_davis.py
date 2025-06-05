import sys, os, json, time, subprocess, pathlib
from pathlib import Path
from davis2017.davis import DAVIS
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm
import torch

# ── USER‐CONFIGURABLE PATHS ──────────────────────────────────────────────────
DAVIS_ROOT  = Path("./data/DAVIS")          # ← point this at your DAVIS folder
OUT_DIR     = Path("../davis2017eval/sam2_preds_MA_0.15_gray")           # ← where we’ll write out PNGs
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── STEP 1: load the DAVIS “val” split (semi‐supervised task) ───────────────────
ds = DAVIS(str(DAVIS_ROOT), task="semi-supervised", subset="val", resolution="480p")
print(f"Loaded {len(ds.sequences)} validation sequences")

# ── STEP 2: build SAM 2 video predictor ─────────────────────────────────────────
from sam2.build_sam import build_sam2_video_predictor

device = "cuda:3" if torch.cuda.is_available() else "cpu"
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"   # ← adjust if needed
model_cfg       = "configs/sam2.1/sam2.1_hiera_l.yaml"   # ← adjust if needed

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
predictor.to(device)   # make sure model is on CUDA if available

# ── STEP 3: for each DAVIS sequence, run SAM 2 and save outputs ───────────────
for seq in ds.sequences:
    print(f"\n=== Processing sequence: {seq} ===")

    # 3a) directories of images & GT masks for this sequence
    img_dir  = DAVIS_ROOT / "JPEGImages"  / "480p" / seq
    mask_dir = DAVIS_ROOT / "Annotations" / "480p" / seq

    img_paths  = sorted(img_dir.glob("*.jpg"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    assert len(img_paths) == len(mask_paths), (
        f"Image/mask count mismatch in {seq}: {len(img_paths)} vs {len(mask_paths)}"
    )

    # 3b) initialize inference state by “loading” this sequence as a video
    #     We pass the *directory* of frames to init_state.  Internally, it will call
    #     `load_video_frames(video_path=video_dir, ...)` and store all frames in memory.
    video_dir = str(img_dir)  # e.g. "./data/davis/DAVIS/JPEGImages/480p/<seq>"
    inference_state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False
    )

# (b) load the single “00000.png” which contains two different colored regions
    rgb = iio.imread(str(mask_dir / "00000.png"))  # shape (H, W, 3)
    H, W, C = rgb.shape
    assert C == 3, "Expected a 3‐channel (RGB) first‐frame mask."

    # (c) find all unique RGB colors except black
    flat = rgb.reshape(-1, 3)                        # shape (H*W, 3)
    uniq_colors = np.unique(flat, axis=0)            # shape (K, 3), where K ≤ (H*W)
    # Remove the black color (0,0,0) if present
    non_black = [tuple(c) for c in uniq_colors if not np.all(c == 0)]
    if len(non_black) == 0:
        raise RuntimeError(f"No non‐black colors found in {seq}/00000.png")

    # (d) for each unique non‐black color, build a 2D boolean mask and register it
    print(f"Found {len(non_black)} unique non‐black colors in {seq}/00000.png")
    for idx, color in enumerate(non_black):
        # color is something like (200, 0, 0) or (0, 200, 0)
        R, G, B = color
        # build a binary mask: True where pixel == this color
        bin_mask = np.logical_and.reduce([
            rgb[:, :, 0] == R,
            rgb[:, :, 1] == G,
            rgb[:, :, 2] == B
        ])  # shape (H, W), dtype=bool

        # wrap as torch.bool on the same device as SAM 2
        mask2d = torch.from_numpy(bin_mask).to(device)

        # register this mask as object `idx`
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=idx,  # choose 0,1,2,… per color
            mask=mask2d
        )

    # 3e) now propagate through all frames.  As each new frame is processed,
    #     propagate_in_video yields (frame_idx, [obj_ids], video_res_masks).
    #
    #     We’ll save each mask as “00000.png”, “00001.png”, … under OUT_DIR/<seq>/
    seq_out_dir = OUT_DIR / seq
    seq_out_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, obj_ids, video_res_masks in tqdm(
        predictor.propagate_in_video(inference_state),
        total=len(img_paths)-1,
        desc=f"Propagating {seq}"
    ):
        # # ‣ frame_idx is an integer (1,2,3,…).  video_res_masks is a tensor of shape
        # #   (num_objects, H, W).  For DAVIS, num_objects==1.
        # #
        # # ‣ Thresholding has already happened internally; `video_res_masks` is
        # #   a float‐tensor where positive values correspond to predicted “object.”
        # mask_np = (video_res_masks[0].cpu().numpy() > 0.0).astype(np.uint8) * 255

        # # Save with zero‐padded five digits to match DAVIS naming:
        # save_name = f"{frame_idx:05d}.png"
        # save_path = seq_out_dir / save_name
        # iio.imwrite(str(save_path), mask_np)

        # Suppose `video_res_masks` is whatever you get from propagate_in_video:
        #   • If there is only one object, it may be a 2D tensor of shape (H, W)
        #   • If there are multiple objects, it will be a 3D tensor of shape (O, H, W)

        pred_np = video_res_masks.cpu().numpy()   # dtype=float32 or float; # ───────────────────────────────────────────────────────────────
        # Assume you already did:
        #   pred_np = video_res_masks.cpu().numpy()

        # 1) Check how many dimensions `pred_np` has:
        if pred_np.ndim == 2:
            # Case A: single object, shape = (H, W)
            H, W = pred_np.shape
            O = 1
            pred_np = pred_np[np.newaxis, ...]  # -> now shape (1, H, W)

        elif pred_np.ndim == 3:
            # Could be either:
            #  (A) shape = (1, H, W)   ← single object with a leading axis
            #  (B) shape = (O, H, W)   ← multiple objects, no extra channel axis
            if pred_np.shape[0] == 1:
                # Treat as “one‐object” → squeeze to (1, H, W) (already fits our convention)
                O, H, W = pred_np.shape
            else:
                # Multi‐object already: (O, H, W)
                O, H, W = pred_np.shape
            # (no need to reshape because it’s already (O, H, W))

        elif pred_np.ndim == 4:
            # Some SAM 2 builds return (O, 1, H, W). In that case:
            #   • pred_np.shape = (O, 1, H, W)
            #   → we want to drop the “channel” dimension (axis=1).
            O = pred_np.shape[0]
            H = pred_np.shape[2]
            W = pred_np.shape[3]
            pred_np = pred_np[:, 0, :, :]  # now shape (O, H, W)

        else:
            raise RuntimeError(f"Unexpected mask array with ndim={pred_np.ndim}, shape={pred_np.shape}")

        # At this point:
        #   • pred_np is guaranteed to have shape (O, H, W)
        #   • O, H, W are set correctly
        # ───────────────────────────────────────────────────────────────

        # Now you can build your colored output exactly as before:

        colored = np.zeros((H, W, 3), dtype=np.uint8)

        for i in range(O):
            mask_i = (pred_np[i] > 0.0)   # boolean mask (H, W)
            if not mask_i.any():
                continue
            R, G, B = non_black[i]  # the original RGB for object i
            colored[mask_i, 0] = R
            colored[mask_i, 1] = G
            colored[mask_i, 2] = B

        save_name = f"{frame_idx:05d}.png"
        save_path = seq_out_dir / save_name
        iio.imwrite(str(save_path), colored)


    print(f"→ Saved all predicted masks for {seq} in {seq_out_dir}")

print("\nAll sequences processed.")
print(f"Your SAM 2 masks live under: {OUT_DIR}")
