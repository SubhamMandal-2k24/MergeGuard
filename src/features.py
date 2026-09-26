"""
Features for (S1 record, candidate record) pairs -- name similarity,
address similarity, plus a few structural ones like country match and
digit overlap for street numbers/PINs.
"""
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from normalize import normalize_name, normalize_address, token_set, digit_tokens

tqdm.pandas()


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return fuzz.ratio(a, b) / 100.0


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def pair_features(row) -> dict:
    name_a, name_b = row.name_a, row.name_b
    addr_a, addr_b = row.addr_a, row.addr_b

    name_tokens_a, name_tokens_b = token_set(name_a), token_set(name_b)
    addr_tokens_a, addr_tokens_b = token_set(addr_a), token_set(addr_b)
    digits_a, digits_b = digit_tokens(row.raw_addr_a), digit_tokens(row.raw_addr_b)

    return {
        "name_char_ratio": _ratio(name_a, name_b),
        "name_token_jaccard": _jaccard(name_tokens_a, name_tokens_b),
        "name_first_token_match": int(
            bool(name_tokens_a) and bool(name_tokens_b)
            and name_a.split()[0] == name_b.split()[0]
        ),
        "name_len_diff": abs(len(name_a) - len(name_b)),
        "name_exact_match": int(name_a == name_b and bool(name_a)),
        "addr_char_ratio": _ratio(addr_a, addr_b),
        "addr_token_jaccard": _jaccard(addr_tokens_a, addr_tokens_b),
        "addr_digit_jaccard": _jaccard(digits_a, digits_b),
        "addr_len_diff": abs(len(addr_a) - len(addr_b)),
        "country_match": int(row.country_a == row.country_b),
        "combined_token_jaccard": _jaccard(
            name_tokens_a | addr_tokens_a, name_tokens_b | addr_tokens_b
        ),
    }


def build_feature_matrix(pairs_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame()
    feats = pairs_df.progress_apply(pair_features, axis=1, result_type="expand")
    return feats


def make_pair_frame(s1_df: pd.DataFrame, other_df: pd.DataFrame, candidates: dict, chunk_size: int = 20000) -> pd.DataFrame:
    """Builds (S1, candidate) pairs in chunks and concatenates small
    DataFrames, instead of growing one giant list of tens of millions of
    dicts in memory before converting -- that approach exhausted RAM and
    crashed VS Code (OOM) at real dataset scale."""
    if not candidates:
        return pd.DataFrame()

    s1_ids_needed = set(candidates.keys())
    other_ids_needed = set()
    for cand_ids in candidates.values():
        other_ids_needed |= cand_ids

    s1_records = (
        s1_df[s1_df["entity_id"].isin(s1_ids_needed)]
        .set_index("entity_id")
        .to_dict("index")
    )
    other_records = (
        other_df[other_df["entity_id"].isin(other_ids_needed)]
        .set_index("entity_id")
        .to_dict("index")
    )

    # Normalize each unique record once, not once per pair.
    s1_norm = {
        eid: (normalize_name(r["business_name"]), normalize_address(r["business_address"]))
        for eid, r in s1_records.items()
    }
    other_norm = {
        eid: (normalize_name(r["business_name"]), normalize_address(r["business_address"]))
        for eid, r in other_records.items()
    }

    s1_ids = list(candidates.keys())
    chunk_frames = []

    for start in tqdm(range(0, len(s1_ids), chunk_size), desc="building pairs (chunked)"):
        chunk_ids = s1_ids[start:start + chunk_size]
        rows = []
        for s1_id in chunk_ids:
            cand_ids = candidates[s1_id]
            if s1_id not in s1_records or not cand_ids:
                continue
            s1_row = s1_records[s1_id]
            name_a, addr_a = s1_norm[s1_id]
            for cid in cand_ids:
                if cid not in other_records:
                    continue
                o_row = other_records[cid]
                name_b, addr_b = other_norm[cid]
                rows.append(
                    {
                        "source1_entity_id": s1_id,
                        "candidate_entity_id": cid,
                        "name_a": name_a,
                        "name_b": name_b,
                        "addr_a": addr_a,
                        "addr_b": addr_b,
                        "raw_addr_a": s1_row["business_address"],
                        "raw_addr_b": o_row["business_address"],
                        "country_a": s1_row["country"],
                        "country_b": o_row["country"],
                    }
                )
        if rows:
            chunk_frames.append(pd.DataFrame(rows))

    if not chunk_frames:
        return pd.DataFrame()
    return pd.concat(chunk_frames, ignore_index=True)