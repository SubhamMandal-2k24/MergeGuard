"""
Features for (S1 record, candidate record) pairs -- name similarity,
address similarity, plus a few structural ones like country match and
digit overlap for street numbers/PINs.

Using difflib for the ratio since we didn't want to depend on
rapidfuzz/python-Levenshtein being pre-installed. If we add rapidfuzz to
requirements.txt later, fuzz.ratio() would probably beat this a bit.
"""
from difflib import SequenceMatcher

import pandas as pd

from normalize import normalize_name, normalize_address, token_set, digit_tokens


def _ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def pair_features(row) -> dict:
    """row needs name_a/name_b, addr_a/addr_b (normalized), country_a/b,
    and the raw address strings (raw_addr_a/b) since we pull digits before
    normalization strips the punctuation around them."""
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
    """Runs pair_features row by row, returns the numeric feature matrix
    lined up with pairs_df's row order."""
    feats = pairs_df.apply(pair_features, axis=1, result_type="expand")
    return feats


def make_pair_frame(s1_df: pd.DataFrame, other_df: pd.DataFrame, candidates: dict) -> pd.DataFrame:
    """Turns the {s1_id: {candidate_ids}} dict from blocking into one row
    per (s1_id, candidate_id) pair, with both normalized and raw text
    pulled in so build_feature_matrix can run on it directly."""
    s1_idx = s1_df.set_index("entity_id")
    other_idx = other_df.set_index("entity_id")

    rows = []
    for s1_id, cand_ids in candidates.items():
        if s1_id not in s1_idx.index or not cand_ids:
            continue
        s1_row = s1_idx.loc[s1_id]
        for cid in cand_ids:
            if cid not in other_idx.index:
                continue
            o_row = other_idx.loc[cid]
            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "name_a": normalize_name(s1_row["business_name"]),
                    "name_b": normalize_name(o_row["business_name"]),
                    "addr_a": normalize_address(s1_row["business_address"]),
                    "addr_b": normalize_address(o_row["business_address"]),
                    "raw_addr_a": s1_row["business_address"],
                    "raw_addr_b": o_row["business_address"],
                    "country_a": s1_row["country"],
                    "country_b": o_row["country"],
                }
            )
    return pd.DataFrame(rows)