"""
Training pipeline:
  1. load train data
  2. split by S1 entity (not by row -- don't want one entity's candidates
     split across train/val)
  3. run blocking, check recall against ground truth
  4. label the candidate pairs (positive = actually in ground truth)
  5. train the classifier (LightGBM if installed, otherwise sklearn's
     HistGB so this still runs before pip install)
  6. sweep thresholds on val, pick whatever maximizes macro F_0.5
  7. save model + threshold

Usage (from src/):
    python train.py --data-dir ../dataset
"""
import argparse
import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from utils import read_tsv, load_ground_truth, macro_f_beta, parse_id_list
from blocking import generate_candidates, measure_blocking_recall
from features import make_pair_frame, build_feature_matrix

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def entity_split(s1_ids, val_frac=0.2, seed=42):
    ids = list(s1_ids)
    random.Random(seed).shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])  # train_ids, val_ids


def build_labeled_pairs(s1_df, other_df, candidates, ground_truth):
    """candidates from blocking, ground_truth filtered to this source's ids
    already (caller does that). Labels each pair 1/0 depending on whether
    it's actually in ground truth, returns X, y, and the pair frame (need
    the frame later to group predictions back by S1 entity)."""
    pair_frame = make_pair_frame(s1_df, other_df, candidates)
    if pair_frame.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pair_frame
    X = build_feature_matrix(pair_frame)
    y = pair_frame.apply(
        lambda r: int(r["candidate_entity_id"] in ground_truth.get(r["source1_entity_id"], set())),
        axis=1,
    )
    return X, y, pair_frame


def train_classifier(X, y):
    if HAS_LGB:
        clf = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
        )
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced")
    clf.fit(X, y)
    return clf


def predict_proba(clf, X):
    return clf.predict_proba(X)[:, 1]


def tune_threshold(val_pair_frame, val_probs, ground_truth_full):
    """Sweeps thresholds, builds the predicted match set per S1 entity at
    each one, scores with macro F_0.5, keeps whichever threshold wins.
    ground_truth_full needs to cover every val entity, singletons included --
    an entity with nothing above threshold still needs to be scored (1.0 if
    it's a true singleton, 0.0 otherwise)."""
    val_pair_frame = val_pair_frame.copy()
    val_pair_frame["prob"] = val_probs

    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.30, 0.96, 0.02):
        pred_map = {}
        for s1_id, grp in val_pair_frame[val_pair_frame["prob"] >= t].groupby(
            "source1_entity_id"
        ):
            pred_map[s1_id] = set(grp["candidate_entity_id"])
        score = macro_f_beta(pred_map, ground_truth_full, beta=0.5)
        if score > best_score:
            best_score, best_t = score, t
    return best_t, best_score


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    s1 = read_tsv(os.path.join(data_dir, "train", "train_source1.tsv"))
    s2 = read_tsv(os.path.join(data_dir, "train", "train_source2.tsv"))
    s3 = read_tsv(os.path.join(data_dir, "train", "train_source3.tsv"))
    gt = load_ground_truth(os.path.join(data_dir, "train", "train_ground_truth.tsv"))

    train_ids, val_ids = entity_split(s1["entity_id"].tolist())
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)

    all_X, all_y = [], []

    for other_df, prefix, tag in [(s2, "S2-", "s2"), (s3, "S3-", "s3")]:
        cand_train = generate_candidates(s1_train, other_df)
        gt_this_source = {
            eid: {m for m in ids if m.startswith(prefix)}
            for eid, ids in gt.items()
            if eid in train_ids
        }
        recall = measure_blocking_recall(cand_train, gt_this_source)
        print(f"[{tag}] blocking recall on train split: {recall:.4f}")

        X, y, pf = build_labeled_pairs(s1_train, other_df, cand_train, gt)
        if not X.empty:
            all_X.append(X)
            all_y.append(y)

    X_train = pd.concat(all_X, ignore_index=True)
    y_train = pd.concat(all_y, ignore_index=True)
    print(f"Training rows: {len(X_train)}, positive rate: {y_train.mean():.4f}")

    clf = train_classifier(X_train, y_train)

    # rebuild candidates + features for the val split so we're evaluating
    # on the same blocking behavior we'll actually see at inference time
    val_pair_frames, val_probs = [], []
    for other_df in (s2, s3):
        cand_val = generate_candidates(s1_val, other_df)
        pf_val = make_pair_frame(s1_val, other_df, cand_val)
        if pf_val.empty:
            continue
        Xv = build_feature_matrix(pf_val)
        probs = predict_proba(clf, Xv)
        val_pair_frames.append(pf_val)
        val_probs.append(probs)

    val_pair_frame_all = pd.concat(val_pair_frames, ignore_index=True)
    val_probs_all = np.concatenate(val_probs)

    gt_val = {eid: ids for eid, ids in gt.items() if eid in val_ids}
    best_t, best_score = tune_threshold(val_pair_frame_all, val_probs_all, gt_val)
    print(f"Best threshold: {best_t:.2f}  ->  val macro F_0.5: {best_score:.4f}")

    joblib.dump(clf, os.path.join(out_dir, "matcher_model.joblib"))
    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": float(best_t), "val_f0.5": float(best_score)}, f)
    print(f"Saved model + threshold to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../dataset")
    ap.add_argument("--out-dir", default="../artifacts")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)