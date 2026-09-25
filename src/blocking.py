"""
Candidate generation (blocking) for Source-1 vs Source-2/3.

We use TF-IDF over character n-grams (name + address combined) and pull the
top-K nearest neighbours by cosine similarity, per source. Ran within
country buckets to keep it fast, plus a smaller cross-country pass since
country labels can be noisy and the test set has France, which isn't in
train at all.

Whatever doesn't show up here can never be matched later, so this is the
step that decides our recall ceiling. We check that against
train_ground_truth before doing anything else (see measure_blocking_recall
below) -- if recall here is bad, tuning the classifier won't fix it.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

from normalize import normalize_name, normalize_address

TOP_K_IN_COUNTRY = 15
TOP_K_CROSS_COUNTRY = 5
MAX_CANDIDATES_PER_SOURCE = 20


def _combined_text(df: pd.DataFrame) -> pd.Series:
    # name + address in one string, normalized -- this is what we vectorize
    return (
        df["business_name"].map(normalize_name)
        + " "
        + df["business_address"].map(normalize_address)
    )


def _topk_neighbors(query_vecs, ref_vecs, k):
    # cosine kNN, returns (indices, similarity scores)
    k = min(k, ref_vecs.shape[0])
    if k == 0:
        return np.empty((query_vecs.shape[0], 0), dtype=int), np.empty(
            (query_vecs.shape[0], 0)
        )
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(ref_vecs)
    dist, idx = nn.kneighbors(query_vecs)
    sim = 1 - dist
    return idx, sim


def generate_candidates(source1: pd.DataFrame, other: pd.DataFrame) -> dict:
    """
    Run once for (source1, source2), once for (source1, source3), then
    union the two dicts to get the full candidate set per S1 entity.
    Returns {source1_entity_id: set(other_entity_id, ...)}
    """
    s1_text = _combined_text(source1)
    other_text = _combined_text(other)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    all_text = pd.concat([s1_text, other_text], ignore_index=True)
    vectorizer.fit(all_text)

    s1_vecs = vectorizer.transform(s1_text)
    other_vecs = vectorizer.transform(other_text)

    candidates = {eid: set() for eid in source1["entity_id"]}

    # same-country top-K -- cheap, and country is usually reliable
    for country, s1_group in source1.groupby("country"):
        other_mask = other["country"] == country
        if not other_mask.any():
            continue
        other_group = other[other_mask]
        g_idx = s1_group.index
        q_vecs = s1_vecs[source1.index.get_indexer(g_idx)]
        r_vecs = other_vecs[other.index.get_indexer(other_group.index)]
        idx, sim = _topk_neighbors(q_vecs, r_vecs, TOP_K_IN_COUNTRY)
        other_ids = other_group["entity_id"].values
        for row_pos, eid in enumerate(s1_group["entity_id"].values):
            for col in range(idx.shape[1]):
                candidates[eid].add(other_ids[idx[row_pos, col]])

    # small cross-country pass -- catches cases where country is wrong/missing,
    # and is our only shot at France since train never saw it
    idx, sim = _topk_neighbors(s1_vecs, other_vecs, TOP_K_CROSS_COUNTRY)
    other_ids_all = other["entity_id"].values
    for row_pos, eid in enumerate(source1["entity_id"].values):
        for col in range(idx.shape[1]):
            candidates[eid].add(other_ids_all[idx[row_pos, col]])

    # trim to the cap -- note: this is an unordered set at this point, so the
    # trim isn't "keep the best ones," it's arbitrary. TODO fix by keeping
    # scores through the union step and cutting by score instead, if this
    # cap is actually getting hit a lot on the real data.
    for eid in candidates:
        if len(candidates[eid]) > MAX_CANDIDATES_PER_SOURCE:
            candidates[eid] = set(list(candidates[eid])[:MAX_CANDIDATES_PER_SOURCE])

    return candidates


def measure_blocking_recall(candidates: dict, ground_truth: dict) -> float:
    """Fraction of true matches that actually made it into our candidate
    sets. Run this first, every time blocking changes. If it's not close to
    1.0 the classifier can't save us -- go fix blocking instead."""
    hit, total = 0, 0
    for eid, true_ids in ground_truth.items():
        cand = candidates.get(eid, set())
        for tid in true_ids:
            total += 1
            if tid in cand:
                hit += 1
    return hit / total if total else 1.0