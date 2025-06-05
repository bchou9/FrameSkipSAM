# change_detection.py
import cv2
import numpy as np
import torch

# In change_detection.py, add outside-mask check:
import numpy as np, cv2, torch

SKIP_MAD_THRESHOLD = 0.05

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_grayscale_uint8(frame):
    """
    Handle CHW or HWC, torch or NumPy.  Returns uint8 grayscale.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    # CHW -> HWC if needed
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = np.transpose(frame, (1, 2, 0))
    # Undo SAM2 normalisation to real pixel space
    frame = (frame * std + mean) * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def should_skip_frame(prev_frame, curr_frame, prev_masks=None, threshold: float = SKIP_MAD_THRESHOLD) -> bool:
    prev_gray = _to_grayscale_uint8(prev_frame)
    curr_gray = _to_grayscale_uint8(curr_frame)
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    normalized_diff = diff.astype(np.float32) / 255.0
    
    if prev_masks is not None:
        # Use same mask processing as in _mean_abs_diff
        union_mask = np.zeros_like(prev_gray, dtype=bool)
        for mask in prev_masks:
            if mask.shape != prev_gray.shape:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), 
                    dsize=(prev_gray.shape[1], prev_gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask.astype(bool)
            union_mask |= mask_resized
        
        if union_mask.any():
            # Calculate mean change in masked regions (normalized 0-1)
            masked_mad = normalized_diff[union_mask].mean()
            return masked_mad < threshold
        else:
            # No mask area - treat as no change
            return True
    
    # Fallback to global MAD
    global_mad = normalized_diff.mean()
    return global_mad < threshold