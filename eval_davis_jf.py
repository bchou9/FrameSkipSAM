# eval_davis_jf.py
import os
import cv2
import numpy as np

def compute_JF(pred_mask, gt_mask):
    # Binarize (0/255 → 0/1)
    p = (pred_mask > 127).astype(np.uint8)
    g = (gt_mask   > 127).astype(np.uint8)
    # Intersection & union
    inter = np.logical_and(p, g).sum()
    union = np.logical_or (p, g).sum()
    J = inter / union if union > 0 else 1.0
    # Contour F: morphological gradient
    kernel = np.ones((3,3), np.uint8)
    p_edge = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
    g_edge = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)
    tp = np.logical_and(p_edge, g_edge).sum()
    prec = tp / p_edge.sum() if p_edge.sum() > 0 else 1.0
    rec  = tp / g_edge.sum() if g_edge.sum() > 0 else 1.0
    F = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return J, F

# Paths—adjust these if your layout differs
ROOT = os.path.dirname(__file__)
DAVIS_ROOT = os.path.join(ROOT, "datasets", "DAVIS", "DAVIS2017", "DAVIS")
PRED_ROOT  = os.path.join(ROOT, "predictions", "DAVIS2017")

all_J = 0.0
all_F = 0.0
count = 0

# Iterate sequences
seqs = sorted(os.listdir(os.path.join(DAVIS_ROOT, "Annotations", "480p")))
for seq in seqs:
    gt_dir   = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq)
    pred_dir = os.path.join(PRED_ROOT, seq)
    if not os.path.isdir(pred_dir):
        print(f"Warning: missing prediction folder for sequence {seq}")
        continue

    frames = sorted(os.listdir(gt_dir))
    for fname in frames:
        gt_path = os.path.join(gt_dir, fname)
        pr_path = os.path.join(pred_dir, fname)

        # Read GT and prediction in grayscale
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pr is None:
            print(f"Warning: unable to load {gt_path} or {pr_path}")
            continue

        # Resize pred → GT if shapes differ
        if pr.shape != gt.shape:
            pr = cv2.resize(
                pr,
                dsize=(gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        J, F = compute_JF(pr, gt)
        all_J += J
        all_F += F
        count += 1

mean_J = all_J / count if count > 0 else 0.0
mean_F = all_F / count if count > 0 else 0.0
print(f"DAVIS-2017 (all frames={count}): Mean J = {mean_J:.4f}, Mean F = {mean_F:.4f}")
