# change_detection.py
import cv2
import numpy as np
import torch

# In change_detection.py, add outside-mask check:
import numpy as np, cv2, torch

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

def should_skip_frame(prev_frame, curr_frame, prev_masks=None, threshold: float=0.05) -> bool:
    # (Normalize and grayscale as before)
    prev_gray = _to_grayscale_uint8(prev_frame)
    curr_gray = _to_grayscale_uint8(curr_frame)
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    if prev_masks is not None:
        # Build flat list of 2D masks
        if isinstance(prev_masks, torch.Tensor):
            prev_masks = prev_masks.cpu().numpy()
        mask_list = []
        candidates = prev_masks if isinstance(prev_masks, (list, tuple)) else [prev_masks]
        for m in candidates:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            if m.ndim == 3:
                for k in range(m.shape[0]):
                    mask_list.append(m[k].astype(bool))
            else:
                mask_list.append(m.astype(bool))

        H, W = diff.shape
        # Build union of all object masks
        union_mask = np.zeros((H, W), dtype=bool)
        for mask in mask_list:
            mask2 = mask[0] if (mask.ndim == 3) else mask
            if mask2.shape != (H, W):
                mask2 = cv2.resize(mask2.astype(np.uint8), dsize=(W, H),
                                   interpolation=cv2.INTER_NEAREST)
            mask_bool = mask2.astype(bool)
            union_mask |= mask_bool

        # Check changes outside the union of masks
        if union_mask.any():
            outside_mask = ~union_mask
            if outside_mask.any():
                frac_out = diff[outside_mask].mean() / 255.0
                if frac_out >= threshold:
                    return False  # outside region changed ⇒ do not skip

        # Now check each object region
        for mask in mask_list:
            mask2 = mask[0] if (mask.ndim == 3) else mask
            if mask2.shape != (H, W):
                mask2 = cv2.resize(mask2.astype(np.uint8), dsize=(W, H),
                                   interpolation=cv2.INTER_NEAREST)
            mask_bool = mask2.astype(bool)
            if not mask_bool.any():
                continue
            frac_changed = diff[mask_bool].mean() / 255.0
            if frac_changed >= threshold:
                return False  # object moved significantly ⇒ do not skip
        return True  # no significant change inside or outside masks

    # If no masks given, fallback to global mean diff
    mean_diff = diff.mean() / 255.0
    return mean_diff < threshold
