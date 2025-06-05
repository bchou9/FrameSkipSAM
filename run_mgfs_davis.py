# run_mgfs_davis.py
import os
import cv2
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.change_detection import SKIP_MAD_THRESHOLD

# Path to your DAVIS download
DAVIS_ROOT = os.path.join(os.path.dirname(__file__), "datasets", "DAVIS", "DAVIS2017", "DAVIS")

# Directory where predictions will be saved
OUT_ROOT = os.path.join(os.path.dirname(__file__), "predictions", "DAVIS2017_optical")
os.makedirs(OUT_ROOT, exist_ok=True)

# Initialize the predictor once
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = build_sam2_video_predictor(checkpoint="checkpoints/sam2.1_hiera_large.pt", config_file="configs/sam2.1/sam2.1_hiera_l.yaml").to(device)
predictor.skip_mad_threshold = SKIP_MAD_THRESHOLD 

# Load all 480p sequence names
seq_dir = os.path.join(DAVIS_ROOT, "JPEGImages", "480p")
sequences = sorted(os.listdir(seq_dir))

for seq in sequences:
    seq_img_dir  = os.path.join(seq_dir, seq)
    seq_mask_dir = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq)
    out_seq_dir  = os.path.join(OUT_ROOT, seq)
    os.makedirs(out_seq_dir, exist_ok=True)

    # Initialize SAM2 state
    state = predictor.init_state(seq_img_dir)
    
    
    # original code
    # # Load first-frame ground truth mask and add as prompt
    # mask0 = cv2.imread(os.path.join(seq_mask_dir, "00000.png"), cv2.IMREAD_GRAYSCALE)
    # predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=mask0)
    
    mask0 = cv2.imread(os.path.join(seq_mask_dir, "00000.png"), cv2.IMREAD_GRAYSCALE)
    # If multiple objects are present in the initial mask, add each one separately
    unique_ids = np.unique(mask0)
    for obj_id in unique_ids:
        if obj_id == 0:  # skip background label 0
            continue
        obj_mask = np.where(mask0 == obj_id, 255, 0).astype(np.uint8)
        predictor.add_new_mask(state, frame_idx=0, obj_id=int(obj_id), mask=obj_mask)

    # Run propagation (with frame skipping)
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # original code
        # # masks: Tensor [N_objects, H, W] on GPU
        # mask_np = masks[0].cpu().numpy().astype(np.uint8) * 255
        
        
        # masks: Tensor [N_objects, H, W] on GPU 
        mask_tensor = masks.squeeze(1).cpu().numpy().astype(np.uint8) 
        if mask_tensor.shape[0] == 1:
            # Only one object: make object mask white (255)
            mask_np = mask_tensor[0] * 255
        else:
            # Combine masks for all objects into one label image
            mask_np = np.zeros_like(mask_tensor[0])
            for m_idx, obj_id in enumerate(obj_ids):
                mask_np[mask_tensor[m_idx] > 0] = obj_id
                
        assert mask_np.ndim == 2, f"mask_np wrong shape {mask_np.shape}"
        if mask_np.size == 0:
                print(f"[WARN] Empty mask for frame {frame_idx} – skipping write")
                continue
                
       
        out_path = os.path.join(out_seq_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(out_path, mask_np)

    print(f"Finished sequence {seq}")
