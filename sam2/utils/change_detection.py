# change_detection.py

import cv2
import numpy as np
import torch
from typing import List
from sam2.utils.optical_flow import warp_mask_forward

SKIP_MAD_THRESHOLD = 0.05

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_grayscale_uint8(frame: np.ndarray) -> np.ndarray:
    """
    Handle CHW or HWC, torch or NumPy, BUT assume `frame` is a FLOAT tensor/array
    in [0..1] that was normalized by (mean,std).  Returns uint8 grayscale.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()

    # If CHW (3, H, W), convert to HWC
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # Un-normalize: [0..1] → real‐pixel [0..255]
    frame = (frame * std + mean) * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Convert RGB → grayscale
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def should_skip_frame(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    prev_masks: List[np.ndarray] = None,
    threshold: float = SKIP_MAD_THRESHOLD
) -> bool:
    """
    Decide whether to skip processing `curr_frame`, based on MAD (global or masked).
    - If `prev_masks` is provided (a list of boolean or uint8 masks in “prev_frame” coords),
      warp those masks forward (using dense optical flow) into `curr_frame`. Then compute
      the MAD only inside the union of all warped masks. If that union is empty, treat as
      “no change” → skip (return True).
    - Otherwise, fallback to a full‐frame (global) MAD between prev_gray and curr_gray.
    - Return True if MAD < threshold.
    """
    prev_gray = _to_grayscale_uint8(prev_frame)
    curr_gray = _to_grayscale_uint8(curr_frame)

    # 2) If we have previous masks, warp them forward via optical flow
    if prev_masks is not None and len(prev_masks) > 0:
        h, w = prev_gray.shape
        mask_stack = []
        for m in prev_masks:
            # Resize mask to match prev_gray if needed
            if m.shape != (h, w):
                m_resized = cv2.resize(
                    m.astype(np.uint8),
                    dsize=(w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)
            else:
                m_resized = m.astype(np.uint8)
            mask_stack.append(m_resized.astype(np.float32) / 255.0)

        # Build a Torch tensor (N_obj, H, W) of normalized‐float masks
        masks_np = np.stack(mask_stack, axis=0)  # shape: (N_obj, H, W), float32 in [0..1]
        prev_masks_tensor = torch.from_numpy(masks_np)  # shape: (N_obj, H, W)

        # Warp all masks into the current frame via the optical‐flow routine
        warped_tensor = warp_mask_forward(
            prev_masks_tensor,
            prev_frame,
            curr_frame
        )  # → torch.Tensor, shape (N_obj, H, W) or (1, N_obj, H, W)

        # Convert warped scores → boolean masks again (threshold at 0.5)
        warped_np = warped_tensor.cpu().numpy()
        if warped_np.ndim == 4:
            warped_np = warped_np.squeeze(0)  # now (N_obj, H, W)

        warped_masks: List[np.ndarray] = []
        for i in range(warped_np.shape[0]):
            warped_masks.append((warped_np[i] > 0.5).astype(bool))

        # Compute masked MAD inside the **union** of all warped masks
        union_mask = np.zeros_like(prev_gray, dtype=bool)
        for mask in warped_masks:
            union_mask |= mask

        if union_mask.any():
            diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32) / 255.0
            masked_mad = diff[union_mask].mean()
            return masked_mad < threshold
        else:
            # No warped‐mask pixels → treat as no change → skip
            return True

    # 3) Fallback: global MAD over the entire frame
    diff_full = cv2.absdiff(prev_gray, curr_gray).astype(np.float32) / 255.0
    global_mad = diff_full.mean()
    return global_mad < threshold
