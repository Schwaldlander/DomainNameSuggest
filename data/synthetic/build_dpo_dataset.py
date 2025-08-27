# data/build_dpo_dataset.py
# -*- coding: utf-8 -*-


from __future__ import annotations
import os, sys, json, argparse, random
from typing import List, Dict, Any, Optional
import numpy as np
import regex as re
import torch

# --- robust imports from your project (both module or script execution) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from inference.judge import judge_domains
import argparse, json, random, hashlib, os, yaml, regex as re
from pathlib import Path
from typing import List, Dict, Tuple

SAFE_REFUSAL = "I can’t help with that request, but I can suggest safe, appropriate domain ideas for permitted businesses."

PROMPT_TMPL = """You are a helpful branding assistant. Given a business description, propose a list of short, brandable domain name ideas that follow these rules:
- Up to 12 characters before the TLD (prefer shorter).
- No spaces, emojis, or underscores.
- Avoid numbers and hyphens unless absolutely necessary.
- Prioritize .com, but consider {tlds} when relevant.
- Return 12 unique options, one per line, no extra commentary.
Business: {biz}
"""

# -------------------------------
# Utility: scoring + parsing
# -------------------------------

EMOJI_RE = re.compile(r"\p{Emoji}", re.UNICODE)

def split_root_tld(name: str, tlds: List[str]) -> Tuple[str, str]:
    """Return (root, tld) if matches any tld; else ('', '').
    Accepts both 'foo.com' and 'foo'+'com' concatenations like 'foo.com' or 'foocom' if tld given with dot."""
    n = name.strip().lower()
    # Exact dot match wins
    for t in sorted(tlds, key=len, reverse=True):
        if n.endswith(t):
            root = n[: -len(t)]
            root = root[:-1] if root.endswith(".") else root
            return root, t
    # Fallback: allow concatenated endings, e.g., 'foocom' (not recommended, but scoreable)
    for t in sorted(tlds, key=len, reverse=True):
        flat = t.replace(".", "")
        if n.endswith(flat):
            root = n[: -len(flat)]
            return root, t
    return "", ""

def judge_domain(name: str, tlds: List[str]) -> Dict:
    """
    Heuristic quality judge for single domain.
    Returns a dict with boolean checks, sub-scores, and an aggregate score in [0, 1].
    """
    original = name
    name = name.strip()
    lower = name.lower()

    root, tld = split_root_tld(lower, tlds)
    has_tld = bool(tld)
    root_clean = root

    # Checks (booleans)
    no_space = (" " not in name)
    no_underscore = ("_" not in name)
    no_emoji = (EMOJI_RE.search(name) is None)
    no_digits = (re.search(r"\d", name) is None)
    no_hyphen = ("-" not in root_clean)  # hyphens in root are discouraged
    length_ok = (len(root_clean) > 0 and len(root_clean) <= 12)
    letters_only = bool(re.fullmatch(r"[a-z]+", root_clean)) if root_clean else False

    # TLD preference
    tld_pref_score = 1.0 if tld == ".com" else (0.7 if tld in tlds else 0.0)

    # Subscores (0/1 style)
    checks = {
        "has_tld": has_tld,
        "length_ok": length_ok,
        "no_space": no_space,
        "no_underscore": no_underscore,
        "no_emoji": no_emoji,
        "no_digits": no_digits,
        "no_hyphen": no_hyphen,
        "letters_only": letters_only,
        "tld": tld or "",
        "root_len": len(root_clean) if root_clean else 0
    }

    # Weighted aggregate (tuned heuristically; sums to 1.0)
    # Rule weights
    w = {
        "length_ok": 0.18,
        "no_space": 0.10,
        "no_underscore": 0.10,
        "no_emoji": 0.10,
        "no_digits": 0.12,
        "no_hyphen": 0.12,
        "letters_only": 0.08,
        "has_tld": 0.05,
        "tld_pref": 0.15
    }

    agg = (
        (1.0 if length_ok else 0.0) * w["length_ok"] +
        (1.0 if no_space else 0.0) * w["no_space"] +
        (1.0 if no_underscore else 0.0) * w["no_underscore"] +
        (1.0 if no_emoji else 0.0) * w["no_emoji"] +
        (1.0 if no_digits else 0.0) * w["no_digits"] +
        (1.0 if no_hyphen else 0.0) * w["no_hyphen"] +
        (1.0 if letters_only else 0.0) * w["letters_only"] +
        (1.0 if has_tld else 0.0) * w["has_tld"] +
        tld_pref_score * w["tld_pref"]
    )

    return {
        "name": original,
        "checks": checks,
        "subscores": {
            "tld_pref_score": tld_pref_score
        },
        "aggregate": round(float(agg), 4)
    }

def score_list(domains: List[str], tlds: List[str]) -> Dict:
    scored = [judge_domain(d, tlds) for d in domains]
    agg = round(sum(d["aggregate"] for d in scored) / max(1, len(scored)), 4)
    return {"aggregate": agg, "domains": scored}

# -------------------------------
# Generation utilities
# -------------------------------

def clean(name):
    return name.strip()

def mk_good_domains(biz, tlds):
    base = re.sub(r"[^a-zA-Z]", "", biz.split()[0]).lower() or "brand"
    syllables = ["ly","io","go","ify","ster","able","loop","nest","zen","hub","lab","luma","nexa","meta","grid"]
    names = set()
    for _ in range(80):
        stem = base[:6] + random.choice(syllables)
        stem = re.sub(r"[^a-z]", "", stem)[:12]
        tld = random.choice(tlds)
        names.add(stem + tld)
        if len(names) >= 16:
            break
    # Prefer .com if present by nudging selection
    sorted_names = sorted(names, key=lambda n: (0 if n.endswith(".com") else 1, len(n)))
    return list(sorted_names)[:12]

def mk_bad_domains(biz, tlds):
    # Violates constraints: too long, hyphens, digits, spaces, underscores
    base = re.sub(r"\s+", "-", biz.lower() if biz else "example")
    names = set()
    for i in range(40):
        n = f"{base}-{i:02d}{random.choice(['_',' '])}{random.choice(['x','99','-'])}{random.choice(tlds)}"
        names.add(n)
        if len(names) >= 12:
            break
    return list(names)[:12]

# -------------------------------
# DPO pair constructors
# -------------------------------

def dpo_pair_for_biz(biz: str, tlds: List[str], category: str = None) -> Dict:
    """
    Produces a DPO pair: chosen (good list) vs rejected (bad list) for a safe business prompt.
    """
    sys_prompt = "You refuse unsafe requests and generate safe, brandable domains."
    user = PROMPT_TMPL.format(biz=biz, tlds=",".join(tlds))
    good = mk_good_domains(biz, tlds)
    bad = mk_bad_domains(biz, tlds)

    chosen_text = "\n".join(good)
    rejected_text = "\n".join(bad)

    return {
        "format": "dpo",
        "system": sys_prompt,
        "prompt": user,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "scores": {
            "chosen": score_list(good, tlds),
            "rejected": score_list(bad, tlds)
        },
        "meta": {
            "type": "business",
            "category": category or "",
        }
    }

def dpo_pair_for_refusal(unsafe_prompt: str, tlds: List[str]) -> Dict:
    """
    DPO pair for safety: chosen = refusal, rejected = giving domains for unsafe prompt.
    """
    sys_prompt = "You must refuse inappropriate or harmful requests."
    user = f"Please generate domain ideas for: {unsafe_prompt}"
    # Make 'rejected' intentionally violate policies and style guidelines
    bad = mk_bad_domains(unsafe_prompt, tlds)
    rejected_text = "\n".join(bad)

    return {
        "format": "dpo",
        "system": sys_prompt,
        "prompt": user,
        "chosen": SAFE_REFUSAL,
        "rejected": rejected_text,
        "scores": {
            # There are no domains in the refusal text; use empty with aggregate 1.0 for chosen (policy),
            # and score the rejected domains as usual.
            "chosen": {"aggregate": 1.0, "domains": []},
            "rejected": score_list(bad, tlds)
        },
        "meta": {
            "type": "refusal"
        }
    }

# -------------------------------
# CLI / Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/synthetic/seed_businesses.yaml")
    ap.add_argument("--safety", default="data/synthetic/safety_lists.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800, help="Approx. number of business DPO pairs (excl. safety pairs).")
    ap.add_argument("--tlds", default=".com,.io,.co,.ai,.app,.net,.org")
    ap.add_argument("--safety_pairs", type=int, default=200, help="Max refusal DPO pairs to include.")
    args = ap.parse_args()

    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)
    with open(args.safety, "r", encoding="utf-8") as f:
        saf = yaml.safe_load(f)
    tlds = [t.strip() for t in args.tlds.split(",") if t.strip().startswith(".")]


    pool = []
    categories = list(seeds.keys())

    # Business DPO pairs
    while len([x for x in pool if x["meta"]["type"] == "business"]) < args.n:
        cat = random.choice(categories)
        biz = random.choice(seeds.get(cat, ["Acme"]))
        pool.append(dpo_pair_for_biz(biz, tlds, category=cat))

    # Refusal DPO pairs (safety)
    cnt_refusal = 0
    for kw in saf.get("banned_keywords", []):
        if cnt_refusal >= args.safety_pairs:
            break
        pool.append(dpo_pair_for_refusal(f"{kw} website", tlds))
        cnt_refusal += 1
        if cnt_refusal >= args.safety_pairs:
            break
        pool.append(dpo_pair_for_refusal(f"{kw} marketplace", tlds))
        cnt_refusal += 1

    # Write JSONL
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as w:
        for row in pool:
            w.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Meta
    sha = hashlib.sha256(("".join(sorted(str(x["prompt"]) + str(x["chosen"]) + str(x["rejected"]) for x in pool))).encode()).hexdigest()[:12]
    meta = {
        "count": len(pool),
        "hash": sha,
        "tlds": tlds,
        "dpo_pairs": {
            "business": len([x for x in pool if x["meta"]["type"] == "business"]),
            "refusal": len([x for x in pool if x["meta"]["type"] == "refusal"])
        },
        "schema": {
            "format": "dpo",
            "fields": ["system", "prompt", "chosen", "rejected", "scores", "meta"],
            "scores": {
                "aggregate_range": [0.0, 1.0],
                "per_domain": {
                    "checks": ["has_tld","length_ok","no_space","no_underscore","no_emoji","no_digits","no_hyphen","letters_only","tld","root_len"],
                    "subscores": ["tld_pref_score"]
                }
            }
        }
    }
    with open(outp.with_suffix(".meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)
    print(f"Wrote {len(pool)} DPO pairs to {outp} (hash={sha})")

if __name__ == "__main__":
    main()
