"""
Candidate generation (blocking) for Source-1 vs Source-2/3.

TF-IDF over character n-grams (name + address combined), top-K nearest
neighbours by cosine similarity, per source. Same-country buckets to keep
it fast, plus a smaller cross-country pass since country labels can be
noisy and the test set has France, which isn't in train at all.

Whatever doesn't show up here can never be matched later, so this step
decides our recall ceiling -- check that against train_ground_truth before
touching anything else (measure_blocking_recall below).

UPDATE (contest organizers, after start): candidate_pairs.tsv is scored
directly now, not just used to audit recall -- a smaller candidate set per
S1 entity ranks higher, on top of the leaderboard score. So this isn't
just "maximize recall" anymore, it's recall vs. candidate-set size. That's
why candidates now carry their similarity score through the whole
pipeline instead of just being a plain set -- we keep the best-scoring
ones and drop weak filler, instead of always keeping a fixed top-K
regardless of how weak the match is.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

from normalize import normalize_name, normalize_address

TOP_K_IN_COUNTRY = 15
TOP_K_CROSS_COUNTRY = 5
MAX_CANDIDATES_PER_SOURCE = 20

# Minimum cosine similarity to even count as a candidate. This is the main
# lever for the recall-vs-candidate-size tradeoff -- raise it to shrink
# candidate sets (better for the new scoring), lower it if recall drops too
# much on real data. 0.15 is a conservative starting point, not tuned yet.
MIN_SIMILARITY = 0.15


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


def _add_scored(scored: dict, eid, other_id, score):
    # keep the higher score if the same pair shows up in both passes
    if other_id not in scored[eid] or score > scored[eid][other_id]:
        scored[eid][other_id] = score


def generate_candidates(source1: pd.DataFrame, other: pd.DataFrame) -> dict:
    """
    Run once for (source1, source2), once for (source1, source3), then
    union the two dicts to get the full candidate set per S1 entity.
    Returns {source1_entity_id: set(other_entity_id, ...)}

    Internally keeps similarity scores so trimming to the cap keeps the
    strongest candidates, not an arbitrary slice of a set.
    """
    s1_text = _combined_text(source1)
    other_text = _combined_text(other)

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    all_text = pd.concat([s1_text, other_text], ignore_index=True)
    vectorizer.fit(all_text)

    s1_vecs = vectorizer.transform(s1_text)
    other_vecs = vectorizer.transform(other_text)

    scored = {eid: {} for eid in source1["entity_id"]}

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
                if sim[row_pos, col] >= MIN_SIMILARITY:
                    _add_scored(scored, eid, other_ids[idx[row_pos, col]], sim[row_pos, col])

    # small cross-country pass -- catches cases where country is wrong/missing,
    # and is our only shot at France since train never saw it
    idx, sim = _topk_neighbors(s1_vecs, other_vecs, TOP_K_CROSS_COUNTRY)
    other_ids_all = other["entity_id"].values
    for row_pos, eid in enumerate(source1["entity_id"].values):
        for col in range(idx.shape[1]):
            if sim[row_pos, col] >= MIN_SIMILARITY:
                _add_scored(scored, eid, other_ids_all[idx[row_pos, col]], sim[row_pos, col])

    # trim to the cap by score now, not by arbitrary set order -- keep the
    # MAX_CANDIDATES_PER_SOURCE strongest matches per entity
    candidates = {}
    for eid, id_scores in scored.items():
        ranked = sorted(id_scores.items(), key=lambda kv: kv[1], reverse=True)
        candidates[eid] = set(oid for oid, _ in ranked[:MAX_CANDIDATES_PER_SOURCE])

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


def candidate_set_stats(candidates: dict) -> dict:
    """Avg/median/max candidates per S1 entity. Now that candidate_pairs.tsv
    is scored directly (smaller candidate sets rank higher), check this
    alongside recall every time blocking changes -- it's the other half of
    the tradeoff, not just a nice-to-have number."""
    sizes = [len(v) for v in candidates.values()]
    if not sizes:
        return {"avg": 0.0, "median": 0.0, "max": 0}
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    median = (
        sizes_sorted[n // 2]
        if n % 2 == 1
        else (sizes_sorted[n // 2 - 1] + sizes_sorted[n // 2]) / 2
    )
    return {"avg": sum(sizes) / n, "median": median, "max": max(sizes)}