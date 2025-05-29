# sam2/utils/optical_flow.py

import cv2
import numpy as np
import torch
from typing import Union

def warp_mask_forward(
    prev_masks: torch.Tensor,
    prev_frame: Union[np.ndarray, torch.Tensor],
    curr_frame: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Warp previous mask(s) to the current frame using dense optical flow.
    - prev_masks: (N_obj,H,W) or (1,N_obj,H,W) tensor of mask scores.
    - prev_frame, curr_frame: either HxWx3 numpy uint8 or 3xHxW tensor in [0,1] or [0,255].
    Returns a tensor of same shape as prev_masks.
    """

    # 1) Extract mask array and its resolution
    if prev_masks.ndim == 4:         # (1, N_obj, H, W)
        masks_np = prev_masks[0].cpu().numpy()
    else:                            # (N_obj, H, W)
        masks_np = prev_masks.cpu().numpy()
    n_obj, mask_h, mask_w = masks_np.shape

    # 2) Convert any input to HxWx3 uint8 RGB
    def to_rgb_uint8(frame):
        if isinstance(frame, torch.Tensor):
            arr = frame.detach().cpu().numpy()
            # If normalized floats in [0,1], scale up
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
        else:
            arr = frame
        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
        # Single-channel -> replicate to 3
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        # Ensure dtype
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    prev_rgb = to_rgb_uint8(prev_frame)
    curr_rgb = to_rgb_uint8(curr_frame)

    # 3) Resize frames to mask resolution
    prev_rgb = cv2.resize(prev_rgb, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)
    curr_rgb = cv2.resize(curr_rgb, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR)

    # 4) Convert resized to grayscale
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    # 5) Compute optical flow at mask resolution
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # 6) Build remap grids
    h, w = mask_h, mask_w
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)

    # 7) Warp each object mask
    warped_np = np.zeros_like(masks_np, dtype=np.float32)
    for i in range(n_obj):
        warped_np[i] = cv2.remap(
            masks_np[i].astype(np.float32),
            map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # 8) Convert back to tensor
    warped_tensor = torch.from_numpy(warped_np)
    if prev_masks.ndim == 4:
        warped_tensor = warped_tensor.unsqueeze(0)
    return warped_tensor
