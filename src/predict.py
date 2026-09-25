"""
Runs the trained model on the test set and writes both output files:
  output/matching_results.tsv   -- this is the one that actually gets scored
  output/candidate_pairs.tsv    -- our full candidate set, for the blocking audit

Usage (from src/):
    python predict.py --data-dir ../dataset --artifacts-dir ../artifacts --out-dir ../output
"""
import argparse
import json
import os

import joblib
import pandas as pd

from utils import read_tsv, write_tsv, join_id_list
from blocking import generate_candidates
from features import make_pair_frame, build_feature_matrix


def main(data_dir, artifacts_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    s1 = read_tsv(os.path.join(data_dir, "test", "test_source1.tsv"))
    s2 = read_tsv(os.path.join(data_dir, "test", "test_source2.tsv"))
    s3 = read_tsv(os.path.join(data_dir, "test", "test_source3.tsv"))

    clf = joblib.load(os.path.join(artifacts_dir, "matcher_model.joblib"))
    with open(os.path.join(artifacts_dir, "threshold.json")) as f:
        threshold = json.load(f)["threshold"]

    all_pair_frames = []
    for other_df in (s2, s3):
        cand = generate_candidates(s1, other_df)
        pf = make_pair_frame(s1, other_df, cand)
        if pf.empty:
            continue
        X = build_feature_matrix(pf)
        pf = pf.copy()
        pf["prob"] = clf.predict_proba(X)[:, 1]
        all_pair_frames.append(pf)

    full = pd.concat(all_pair_frames, ignore_index=True) if all_pair_frames else pd.DataFrame(
        columns=["source1_entity_id", "candidate_entity_id", "prob"]
    )

    # candidate_pairs.tsv -- every candidate we considered, one row per S1 entity
    cand_rows = []
    for eid in s1["entity_id"]:
        ids = full.loc[full["source1_entity_id"] == eid, "candidate_entity_id"]
        cand_rows.append({"source1_entity_id": eid, "candidate_entity_ids": join_id_list(set(ids))})
    write_tsv(pd.DataFrame(cand_rows), os.path.join(out_dir, "candidate_pairs.tsv"))

    # matching_results.tsv -- only what cleared the threshold, but every S1
    # entity still needs a row (empty string if nothing cleared)
    match_rows = []
    above = full[full["prob"] >= threshold]
    for eid in s1["entity_id"]:
        ids = above.loc[above["source1_entity_id"] == eid, "candidate_entity_id"]
        match_rows.append({"source1_entity_id": eid, "matched_entity_ids": join_id_list(set(ids))})
    write_tsv(pd.DataFrame(match_rows), os.path.join(out_dir, "matching_results.tsv"))

    print(f"Wrote {len(match_rows)} rows to matching_results.tsv and candidate_pairs.tsv in {out_dir}")
    print("Now run utils/validate_submission.py against these before you submit.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../dataset")
    ap.add_argument("--artifacts-dir", default="../artifacts")
    ap.add_argument("--out-dir", default="../output")
    args = ap.parse_args()
    main(args.data_dir, args.artifacts_dir, args.out_dir)