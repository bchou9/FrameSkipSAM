# eval_davis_jf.py

import os
import cv2
import numpy as np
import argparse

def compute_J(p, g):
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return inter / union if union > 0 else 1.0

def compute_F(p, g, tol=3):
    kernel = np.ones((3,3), np.uint8)
    p_edge = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel)
    g_edge = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)

    dt_gt = cv2.distanceTransform((1 - g_edge).astype(np.uint8),
                                  cv2.DIST_L2, 5)
    dt_pr = cv2.distanceTransform((1 - p_edge).astype(np.uint8),
                                  cv2.DIST_L2, 5)

    p_pts = np.where(p_edge > 0)
    if p_pts[0].size == 0:
        prec = 1.0
    else:
        prec = np.mean(dt_gt[p_pts] <= tol)

    g_pts = np.where(g_edge > 0)
    if g_pts[0].size == 0:
        rec = 1.0
    else:
        rec = np.mean(dt_pr[g_pts] <= tol)

    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def compute_JF(pr_mask, gt_mask, tol=3):
    pr = (pr_mask > 127).astype(np.uint8)
    gt = (gt_mask > 127).astype(np.uint8)
    J = compute_J(pr, gt)
    F = compute_F(pr, gt, tol)
    return J, F

def eval_predictions(gt_root, pred_root, val_seqs):
    per_seq_J   = {}
    per_seq_F   = {}
    seq_js, seq_fs = [], []

    for seq in val_seqs:
        gt_dir   = os.path.join(gt_root, "Annotations", "480p", seq)
        pr_dir   = os.path.join(pred_root, seq)
        if not os.path.isdir(gt_dir) or not os.path.isdir(pr_dir):
            print(f"[WARN] Missing {seq} in {pred_root}, skipping")
            continue

        js, fs = [], []
        for fname in sorted(os.listdir(gt_dir)):
            gt_path = os.path.join(gt_dir, fname)
            pr_path = os.path.join(pr_dir, fname)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pr = cv2.imread(pr_path, cv2.IMREAD_GRAYSCALE)
            if gt is None or pr is None:
                continue
            if gt.sum() == 0:
                continue
            if pr.shape != gt.shape:
                pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            J, F = compute_JF(pr, gt, tol=3)
            js.append(J)
            fs.append(F)

        if not js:
            continue
        meanJ = sum(js) / len(js)
        meanF = sum(fs) / len(fs)
        per_seq_J[seq] = meanJ
        per_seq_F[seq] = meanF
        seq_js.append(meanJ)
        seq_fs.append(meanF)

    overall_J = sum(seq_js) / len(seq_js) if seq_js else 0.0
    overall_F = sum(seq_fs) / len(seq_fs) if seq_fs else 0.0
    return per_seq_J, per_seq_F, overall_J, overall_F

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gt_root",     required=True,
                   help="DAVIS root (contains Annotations/480p & ImageSets/2017/val.txt)")
    p.add_argument("--pred1",       required=True,
                   help="Baseline predictions root (subfolders per sequence)")
    p.add_argument("--pred2",       required=True,
                   help="MGFS predictions root")
    args = p.parse_args()

    # load val set
    val_txt = os.path.join(args.gt_root, "ImageSets", "2017", "val.txt")
    with open(val_txt) as f:
        val_seqs = [l.strip() for l in f if l.strip()]

    # evaluate both
    j1, f1, oJ1, oF1 = eval_predictions(args.gt_root, args.pred1, val_seqs)
    j2, f2, oJ2, oF2 = eval_predictions(args.gt_root, args.pred2, val_seqs)

    print(f"{'Sequence':20s} |  Baseline J   F  |  MGFS J   F")
    print("-"*55)
    for seq in val_seqs:
        if seq in j1 and seq in j2:
            print(f"{seq:20s} |  {j1[seq]:.4f}   {f1[seq]:.4f}  |  "
                  f"{j2[seq]:.4f}   {f2[seq]:.4f}")
    print("-"*55)
    print(f"{'Mean':20s} |  {oJ1:.4f}   {oF1:.4f}  |  {oJ2:.4f}   {oF2:.4f}")
