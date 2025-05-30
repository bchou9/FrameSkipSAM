# eval_davis_jf.py
import os
import cv2
import numpy as np

def compute_J(p, g):
    # Intersection over Union
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union > 0 else 1.0

def compute_F(p, g, tol=3):
    """
    Compute the DAVIS-style contour accuracy F:
     - Extract binary edges (morphological gradient).
     - Compute distance transform on GT edges and prediction edges.
     - Count how many p-edge pixels lie within tol of any g-edge (precision),
       and vice versa for recall.
    """
    # 1) edges via morphological gradient
    kernel = np.ones((3,3), np.uint8)
    p_edge = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
    g_edge = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)

    # 2) distance maps
    #   dt_gt: for each pixel, distance to nearest g_edge==1
    #   dt_pr: for each pixel, distance to nearest p_edge==1
    # We invert edges: background=0, edge=1, so do distanceTransform on background.
    dt_gt = cv2.distanceTransform((1 - g_edge).astype(np.uint8), 
                                  distanceType=cv2.DIST_L2, 
                                  maskSize=5)
    dt_pr = cv2.distanceTransform((1 - p_edge).astype(np.uint8), 
                                  distanceType=cv2.DIST_L2, 
                                  maskSize=5)

    # 3) precision: fraction of p_edge pixels within tol of any GT edge
    p_pts = np.where(p_edge > 0)
    if p_pts[0].size == 0:
        precision = 1.0
    else:
        dists = dt_gt[p_pts]
        precision = np.mean(dists <= tol)

    # 4) recall: fraction of g_edge pixels within tol of any pred edge
    g_pts = np.where(g_edge > 0)
    if g_pts[0].size == 0:
        recall = 1.0
    else:
        dists = dt_pr[g_pts]
        recall = np.mean(dists <= tol)

    # 5) F-score
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def compute_JF(pred_mask, gt_mask, tol=3):
    # Binarize (0/255 → 0/1)
    pr = (pred_mask > 127).astype(np.uint8)
    gt = (gt_mask   > 127).astype(np.uint8)
    J = compute_J(pr, gt)
    F = compute_F(pr, gt, tol=tol)
    return J, F

# Paths — adjust if needed
ROOT       = os.path.dirname(__file__)
DAVIS_ROOT = os.path.join(ROOT, "datasets", "DAVIS", "DAVIS2017", "DAVIS")
PRED_ROOT  = os.path.join(ROOT, "predictions", "DAVIS2017")

# Load val sequences
val_list = os.path.join(DAVIS_ROOT, "ImageSets", "2017", "val.txt")
with open(val_list, "r") as f:
    val_seqs = [line.strip() for line in f if line.strip()]

seq_js = []
seq_fs = []

for seq in val_seqs:
    gt_dir   = os.path.join(DAVIS_ROOT, "Annotations", "480p", seq)
    pred_dir = os.path.join(PRED_ROOT, seq)
    if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
        print(f"[WARN] Missing GT or preds for sequence {seq}, skipping.")
        continue

    per_j = []
    per_f = []

    for fname in sorted(os.listdir(gt_dir)):
        gt_path = os.path.join(gt_dir, fname)
        pr_path = os.path.join(pred_dir, fname)

        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
        if gt is None or pr is None:
            print(f"[WARN] Could not load {seq}/{fname}, skipping frame.")
            continue

        # Skip empty GT frames
        if gt.sum() == 0:
            continue

        # Resize prediction to GT if needed
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        J, F = compute_JF(pr, gt, tol=3)
        per_j.append(J)
        per_f.append(F)

    if not per_j:
        print(f"[WARN] No valid frames for sequence {seq}.")
        continue

    seq_j = sum(per_j) / len(per_j)
    seq_f = sum(per_f) / len(per_f)
    seq_js.append(seq_j)
    seq_fs.append(seq_f)
    print(f"Sequence {seq:20s} → J = {seq_j:.4f}, F = {seq_f:.4f}")

# Overall means
mean_J = sum(seq_js) / len(seq_js)
mean_F = sum(seq_fs) / len(seq_fs)
print(f"\nDAVIS-2017 val (30 seqs): Mean J = {mean_J:.4f}, Mean F = {mean_F:.4f}")
