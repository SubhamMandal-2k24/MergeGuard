"""
IO helpers plus our own copy of the F_0.5 scorer, so we can check our
score locally before spending one of the 5 daily submissions.
"""
import pandas as pd


def read_tsv(path):
    # everything here is tab-separated -- addresses/ID lists have commas
    # in them so the default comma separator would silently break
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(df, path):
    df.to_csv(path, sep="\t", index=False)


def parse_id_list(s):
    """'S2-00047,S2-00193,S3-00812' -> {'S2-00047','S2-00193','S3-00812'}
    empty string means singleton -> empty set"""
    s = (s or "").strip()
    if not s:
        return set()
    return set(x.strip() for x in s.split(",") if x.strip())


def join_id_list(ids):
    # sorted so output is deterministic across runs
    return ",".join(sorted(ids))


def f_beta_entity(pred_ids: set, true_ids: set, beta: float = 0.5) -> float:
    """F_beta for one S1 entity, matching the spec exactly:
      - true singleton (empty) + we predicted empty -> 1.0
      - true singleton + we predicted a match        -> 0.0 (false merge)
      - true has matches + we predicted nothing       -> 0.0 (recall 0)
      - otherwise normal precision/recall F_beta
    """
    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0
    tp = len(pred_ids & true_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def macro_f_beta(pred_map: dict, true_map: dict, beta: float = 0.5) -> float:
    """Averages f_beta_entity over every entity in true_map -- this is the
    macro-average the spec asks for. Any entity we didn't predict anything
    for just gets treated as an empty prediction."""
    scores = []
    for eid, true_ids in true_map.items():
        pred_ids = pred_map.get(eid, set())
        scores.append(f_beta_entity(pred_ids, true_ids, beta))
    return sum(scores) / len(scores) if scores else 0.0


def load_ground_truth(path):
    """-> {source1_entity_id: set(matched_ids)}"""
    df = read_tsv(path)
    return {
        row.source1_entity_id: parse_id_list(row.matched_entity_ids)
        for row in df.itertuples()
    }