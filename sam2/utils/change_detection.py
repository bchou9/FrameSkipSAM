# change_detection.py
import cv2
import numpy as np
import torch

def should_skip_frame(prev_frame, curr_frame, prev_masks=None, threshold: float = 0.05) -> bool:
    """
    Decide whether the change between prev_frame and curr_frame is small enough to skip segmentation.
    The inputs can be PyTorch tensors (CHW normalized) or NumPy arrays.  threshold is in [0,1].
    Returns True if mean pixel change < threshold * max_change.
    """
    # Convert tensors to NumPy, CHW->HWC
    if isinstance(prev_frame, torch.Tensor):
        prev_frame = prev_frame.cpu().numpy()
    if isinstance(curr_frame, torch.Tensor):
        curr_frame = curr_frame.cpu().numpy()
    if prev_frame.ndim == 3:
        prev_frame = np.transpose(prev_frame, (1, 2, 0))
    if curr_frame.ndim == 3:
        curr_frame = np.transpose(curr_frame, (1, 2, 0))

    # Undo normalization: images were normalized by mean/std and scaled [0,1]
    # Convert back to [0,255] for difference calculation
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    prev_img = (prev_frame * std + mean) * 255.0
    curr_img = (curr_frame * std + mean) * 255.0
    prev_img = np.clip(prev_img, 0, 255).astype(np.uint8)
    curr_img = np.clip(curr_img, 0, 255).astype(np.uint8)

    # Compute grayscale difference
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    
    if prev_masks is not None:
        if prev_masks is not None:
            if isinstance(prev_masks, torch.Tensor):
                prev_masks = prev_masks.cpu().numpy()
            
            # Build a *flat* list of 2-D NumPy masks (one per object)
            mask_list = []
            # Always work with an iterable first
            candidate_masks = (
                prev_masks if isinstance(prev_masks, (list, tuple)) else [prev_masks]
            )
            for m in candidate_masks:
                # tensor → NumPy
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                # If mask is 3-D (N, H, W) split it into N separate 2-D masks
                if m.ndim == 3:
                    for k in range(m.shape[0]):
                        mask_list.append(m[k].astype(bool))           
                else:  # already 2-D
                    mask_list.append(m.astype(bool))
                
        # # Ensure prev_masks is a NumPy array
        # if isinstance(prev_masks, torch.Tensor):
        #     prev_masks = prev_masks.cpu().numpy()
        # # If prev_masks is 2D, convert to a list for uniform processing
        # if prev_masks.ndim == 2:
        #     prev_masks = [prev_masks]
        # # Compute MAD for each mask
        # for mask in prev_masks:
        
        H, W = diff.shape

        for mask in mask_list:
            if mask.ndim == 3:          # rare stray (1, H, W)
                mask = mask[0]

            # ── up-sample to diff resolution if needed ──────────────────
            if mask.shape != (H, W):
                mask = cv2.resize(
                    mask.astype(np.uint8),     # keep binary
                    dsize=(W, H),              # (width, height)
                   interpolation=cv2.INTER_NEAREST,
               )

            mask_bool = mask.astype(bool)
            if not mask_bool.any():            # empty mask → ignore
                continue

            frac_changed = diff[mask_bool].mean() / 255.0  # [0,1]
            if frac_changed >= threshold:      # any object moves ⇒ DON’T skip
                return False
        return True  # No significant change in any mask
    else:
        mean_diff = diff.mean() / 255.0  # normalize to [0,1]
        return mean_diff < threshold
    
    # mean_diff = diff.mean() / 255.0  # normalize to [0,1]
    # return mean_diff < threshold
