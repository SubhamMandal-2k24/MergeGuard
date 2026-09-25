"""
Cleanup for business_name / business_address before we compare anything.
Mainly going after the abbreviation/suffix noise the problem statement
calls out -- Pvt/Private, Ltd/Limited, Rd/Road, etc. -- so the similarity
features are comparing actual content, not spelling variants.

This is just static regex substitution on the fields we were given, not a
lookup against any outside source.
"""
import re

# small, hand-picked list of legal suffixes / common abbreviations -- not
# pulled from anywhere external, just the obvious ones
NAME_REPLACEMENTS = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\binc\b": "incorporated",
    r"\bllp\b": "limited liability partnership",
    r"\bllc\b": "limited liability company",
    r"&": " and ",
}

ADDRESS_REPLACEMENTS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bno\.?\b": "number",
    r"\bnr\b": "near",
    r"&": " and ",
}

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _base_clean(s: str) -> str:
    s = (s or "").lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_name(s: str) -> str:
    s = _base_clean(s)
    for pat, repl in NAME_REPLACEMENTS.items():
        s = re.sub(pat, repl, s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_address(s: str) -> str:
    s = _base_clean(s)
    for pat, repl in ADDRESS_REPLACEMENTS.items():
        s = re.sub(pat, repl, s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def token_set(s: str) -> frozenset:
    return frozenset(s.split()) if s else frozenset()


def digit_tokens(raw_s: str) -> frozenset:
    """Grabs digit runs (street numbers, PIN/ZIP) from the RAW string,
    before normalization strips the punctuation around them -- these
    usually survive rewording even when the rest of the address doesn't."""
    return frozenset(re.findall(r"\d+", raw_s or ""))


def sorted_token_key(s: str) -> str:
    """For catching word-order swaps -- sort the tokens so 'Road MG' and
    'MG Road' land on the same key."""
    return " ".join(sorted(token_set(s)))