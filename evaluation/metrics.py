import regex as re
import numpy as np
from collections import Counter

def brandability_score(domain: str) -> float:
    """
    Heuristic 0..1: short, no digits/hyphens, vowel-consonant balance, no repeats.
    Expects full domain (e.g., "nexaly.com").
    """
    name = domain.split(".")[0]
    if not name:
        return 0.0
    length = len(name)
    # length sweet spot 5..10
    len_score = 1.0 - min(abs(7.0 - length) / 7.0, 1.0) * 0.7
    if re.search(r"[\d_ ]", name):
        len_score -= 0.25
    if "-" in name:
        len_score -= 0.25

    vowels = len(re.findall(r"[aeiou]", name))
    cons = max(1, length - vowels)
    vc_ratio = vowels / cons
    vc_score = 1.0 - min(abs(0.6 - vc_ratio) / 0.6, 1.0) * 0.6

    # penalize triple repeats
    rep_pen = 0.0
    if re.search(r"(.)\1\1", name):
        rep_pen = 0.3

    score = max(0.0, min(1.0, 0.4*len_score + 0.5*vc_score + 0.1*(1.0 - rep_pen)))
    return float(score)

def list_diversity(domains):
    """
    0..1: higher is more diverse (based on unique 3-grams across names).
    """
    grams = []
    for d in domains:
        n = d.split(".")[0]
        grams.extend([n[i:i+3] for i in range(max(0, len(n)-2))])
    if not grams:
        return 0.0
    counts = Counter(grams)
    uniq = len(counts)
    total = len(grams)
    # unique fraction with a diminishing return
    return float(min(1.0, (uniq / max(1,total)) * 2.0))

def valid_domain(d: str, allowed_tlds):
    if not any(d.endswith(t) for t in allowed_tlds):
        return False
    name = d[: -len(next(t for t in allowed_tlds if d.endswith(t)))]
    name = name.rstrip(".")
    if not re.fullmatch(r"[a-z0-9-]{1,63}", name):
        return False
    if name.startswith("-") or name.endswith("-"):
        return False
    if "__" in name or " " in name:
        return False
    return True
