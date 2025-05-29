# change_detection.py
import cv2
import numpy as np
import torch

def should_skip_frame(prev_frame, curr_frame, threshold: float = 0.1) -> bool:
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
    mean_diff = diff.mean() / 255.0  # normalize to [0,1]
    return mean_diff < threshold
