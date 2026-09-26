"""
Candidate generation (blocking) -- REWRITTEN for scale.

The original version used TF-IDF + brute-force nearest-neighbor search,
which is O(n x m) -- fine for a few thousand rows, completely infeasible
once train_source1.tsv turned out to have 2.2 MILLION rows. That's exactly
what the organizers' "blocking must scale, no full pairwise comparison"
warning was about.

New approach: inverted index (the standard way real-world entity
resolution scales). Instead of comparing every record to every record:
  1. Break each record's name+address into tokens.
  2. Build a lookup: token -> which records contain it (per source).
  3. Skip tokens that are too common (appear in a large fraction of
     records) -- these create huge, useless buckets ("private", "road",
     "and") and would blow up the same way full comparison did.
  4. For a given S1 record, only look at records that share at least one
     of its rare/specific tokens. This candidate pool is now small.
  5. Score that (already small) pool with token-overlap + digit-overlap,
     rank, keep the top ones.

This is O(n) to build the index and O(bucket size) per lookup, not
O(n x m). Digit tokens (street numbers, PIN codes) are especially useful
blocking keys here -- they're rare and precise.

Tradeoff vs. the old approach: token-overlap blocking is more sensitive to
heavy typos / totally different wording than TF-IDF character n-grams was.
That's the cost of being able to run at all on millions of rows. If real
recall comes back too low, the first thing to try is raising
MAX_TOKEN_BUCKET_SIZE (allow bigger buckets before dropping a token as a
blocking key). If candidate sets are too large (they matter for ranking
now), raise MIN_OVERLAP_SCORE instead.
"""
from tqdm import tqdm

from collections import defaultdict

import pandas as pd

from normalize import normalize_name, normalize_address, token_set, digit_tokens

# A token whose bucket (how many records contain it) exceeds this absolute
# count is dropped as a blocking key -- it's too common to narrow anything
# down ("private", "road", etc.) and would blow up comparison cost.
# This is an ABSOLUTE cap, not a fraction of dataset size -- a fraction-based
# cutoff behaves wildly differently on 30 rows vs. 2 million rows, which is
# exactly the bug that broke recall on the small test below before this fix.
MAX_TOKEN_BUCKET_SIZE = 800

# Real business records share a LOT of common words (Private, Limited, Road,
# city names) -- far more than synthetic test data did. Unioning every
# token's bucket for a record blew up pool sizes and made this take hours
# on the real 2.2M-row dataset. Fix: only use each record's N *rarest*
# tokens (smallest buckets) to build its candidate pool -- these are the
# most distinctive (street numbers, PINs, unusual words), and this bounds
# pool size hard regardless of how many common words the record also has.
NUM_RARE_TOKENS_FOR_POOL = 7

MAX_CANDIDATES_PER_SOURCE = 35
MIN_OVERLAP_SCORE = 0.05


def _record_tokens(row):
    """All blocking-relevant tokens for one record: name tokens, address
    tokens, and digit tokens from the raw address (street numbers, PINs --
    high-value, low-frequency blocking keys)."""
    name = normalize_name(row["business_name"])
    addr = normalize_address(row["business_address"])
    toks = token_set(name) | token_set(addr)
    digits = digit_tokens(row["business_address"])
    # prefix digits so they don't collide with a name/address token that
    # happens to be the same string
    return toks | {f"#{d}" for d in digits}


def _build_index(df: pd.DataFrame):
    """
    Returns:
      token_to_ids: {token: set(entity_id, ...)} for tokens whose bucket
                    isn't too large to be useful
      record_tokens: {entity_id: set(tokens)} -- every record's own tokens,
                     used later for scoring candidates
    """
    doc_freq = defaultdict(int)
    record_tokens = {}

    for row in df.itertuples(index=False):
        row_d = row._asdict()
        toks = _record_tokens(row_d)
        record_tokens[row_d["entity_id"]] = toks
        for t in toks:
            doc_freq[t] += 1

    token_to_ids = defaultdict(set)
    for eid, toks in record_tokens.items():
        for t in toks:
            if doc_freq[t] <= MAX_TOKEN_BUCKET_SIZE:
                token_to_ids[t].add(eid)

    return token_to_ids, record_tokens


def _overlap_score(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union else 0.0


def generate_candidates(source1: pd.DataFrame, other: pd.DataFrame) -> dict:
    """
    Run once for (source1, source2), once for (source1, source3), then
    union the two result dicts.
    Returns {source1_entity_id: set(other_entity_id, ...)}

    Restricts to same-country pairs by building a separate index per
    country -- this both improves precision and keeps buckets smaller.
    A record with a country not seen on the other side gets no candidates
    from this pass (acceptable for now -- flag if this matters for France
    once we see real country label behavior in the test set).
    """
    candidates = {eid: set() for eid in source1["entity_id"]}

    for country, other_group in other.groupby("country"):
        s1_group = source1[source1["country"] == country]
        if s1_group.empty:
            continue

        token_to_ids, other_tokens = _build_index(other_group)

        for row in tqdm(s1_group.itertuples(index=False), total=len(s1_group), desc=f"blocking {country}"):
            row_d = row._asdict()
            eid = row_d["entity_id"]
            my_tokens = _record_tokens(row_d)

            # only use the RAREST few tokens to build the candidate pool --
            # using every token (including common ones near the cap) is
            # what made this too slow on real data. Digit tokens (#123)
            # tend to be naturally rare and are prioritized implicitly since
            # they usually have the smallest buckets already.
            my_tokens_in_index = [t for t in my_tokens if t in token_to_ids]
            if not my_tokens_in_index:
                continue
            rarest = sorted(my_tokens_in_index, key=lambda t: len(token_to_ids[t]))[:NUM_RARE_TOKENS_FOR_POOL]

            pool = set()
            for t in rarest:
                pool |= token_to_ids[t]

            if not pool:
                continue

            scored = []
            for oid in pool:
                score = _overlap_score(my_tokens, other_tokens[oid])
                if score >= MIN_OVERLAP_SCORE:
                    scored.append((oid, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            candidates[eid] |= {oid for oid, _ in scored[:MAX_CANDIDATES_PER_SOURCE]}

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
    """Avg/median/max candidates per S1 entity. candidate_pairs.tsv is
    scored directly now (smaller candidate sets rank higher), so check this
    alongside recall every time blocking changes."""
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