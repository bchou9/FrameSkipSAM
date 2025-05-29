# eval_davis_jf.py

import os
import cv2
import numpy as np

def compute_JF(pred_mask, gt_mask):
    # Binarize to {0,1}
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask   > 127).astype(np.uint8)
    # Jaccard (IoU)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    J = inter / union if union > 0 else 1.0
    # Boundary F: morphological-gradient edges
    kernel = np.ones((3, 3), np.uint8)
    p_edge = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
    g_edge = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)
    tp = np.logical_and(p_edge, g_edge).sum()
    prec = tp / p_edge.sum() if p_edge.sum() > 0 else 1.0
    rec  = tp / g_edge.sum() if g_edge.sum() > 0 else 1.0
    F = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return J, F

# Adjust these paths if your directory layout differs
ROOT       = os.path.dirname(__file__)
DAVIS_ROOT = os.path.join(ROOT, "datasets", "DAVIS", "DAVIS2017", "DAVIS")
PRED_ROOT  = os.path.join(ROOT, "predictions", "DAVIS2017")

# 1) Load list of val sequences (one sequence name per line)
val_list = os.path.join(DAVIS_ROOT, "ImageSets", "2017", "val.txt")
with open(val_list, "r") as f:
    val_seqs = [line.strip() for line in f if line.strip()]

sequence_js = []
sequence_fs = []

# 2) Iterate each validation sequence
for seq in val_seqs:
    gt_dir   = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq)
    pred_dir = os.path.join(PRED_ROOT, seq)
    if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
        print(f"[WARN] Missing GT or preds for seq {seq}, skipping.")
        continue

    seq_Js = []
    seq_Fs = []

    # 3) Iterate frames in this sequence
    for fname in sorted(os.listdir(gt_dir)):
        gt_path = os.path.join(gt_dir, fname)
        pr_path = os.path.join(pred_dir, fname)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pr is None:
            print(f"[WARN] Could not load {seq}/{fname}, skipping frame.")
            continue

        # 4) Skip genuinely empty GT frames (<no object>)
        if gt.sum() == 0:
            continue

        # 5) Resize pred → GT shape if they differ
        if pr.shape != gt.shape:
            pr = cv2.resize(
                pr,
                dsize=(gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # 6) Compute J & F for this frame
        J, F = compute_JF(pr, gt)
        seq_Js.append(J)
        seq_Fs.append(F)

    if len(seq_Js) == 0:
        print(f"[WARN] No valid frames for sequence {seq}, skipping.")
        continue

    # 7) Sequence‐mean J & F
    J_seq = sum(seq_Js) / len(seq_Js)
    F_seq = sum(seq_Fs) / len(seq_Fs)
    sequence_js.append(J_seq)
    sequence_fs.append(F_seq)
    print(f"Sequence {seq:15s} → J={J_seq:.4f}, F={F_seq:.4f}")

# 8) Final overall means across sequences
Mean_J = sum(sequence_js) / len(sequence_js)
Mean_F = sum(sequence_fs) / len(sequence_fs)
print(f"\nDAVIS-2017 val (30 seqs): Mean J = {Mean_J:.4f}, Mean F = {Mean_F:.4f}")
