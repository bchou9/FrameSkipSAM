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
    prev_gray = _to_grayscale_uint8(prev_frame)
    curr_gray = _to_grayscale_uint8(curr_frame)
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    if prev_masks is not None:
        # Build list of valid 2D masks
        mask_list = []
        for mask in prev_masks:
            if mask.shape != prev_gray.shape:
                mask = cv2.resize(
                    mask.astype(np.uint8), 
                    dsize=(prev_gray.shape[1], prev_gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask = mask.astype(bool)
            mask_list.append(mask)
        
        # Create union of all masks
        union_mask = np.zeros_like(prev_gray, dtype=bool)
        for mask in mask_list:
            union_mask |= mask
        
        # Calculate change metrics
        outside_mask = ~union_mask
        max_change = 0.0
        
        # Check outside region
        if outside_mask.any():
            outside_change = diff[outside_mask].mean() / 255.0
            max_change = max(max_change, outside_change)
        
        # Check inside each object region
        for mask in mask_list:
            if mask.any():
                object_change = diff[mask].mean() / 255.0
                max_change = max(max_change, object_change)
        
        # Skip frame if max change is below threshold
        return max_change < threshold
    
    # Fallback to global MAD if no masks
    global_mad = diff.mean() / 255.0
    return global_mad < threshold
