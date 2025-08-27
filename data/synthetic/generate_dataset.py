# data/synthetic/generate_dataset.py
import argparse, json, random, hashlib, os
from pathlib import Path
import regex as re
import yaml

SAFE_REFUSAL = (
    "I can’t help with that request, but I can suggest safe, appropriate "
    "domain ideas for permitted businesses."
)

PROMPT_TMPL = """You are a helpful branding assistant. Given a business description, propose a list of short, brandable domain name ideas that follow these rules:
- Up to 12 characters before the TLD (prefer shorter).
- No spaces, emojis, or underscores.
- Avoid numbers and hyphens unless absolutely necessary.
- Prioritize .com, but consider {tlds} when relevant.
- Return 12 unique options, one per line, no extra commentary.
Business: {biz}
"""

def _letters(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())

def _ensure_len(name: str, n: int = 12) -> str:
    return name[:n]

def _mk_good_domains(biz: str, tlds):
    base = _letters(biz.split()[0]) or "brand"
    syll = ["ly","io","go","ify","ster","able","loop","nest","zen","hub","lab","luma","nexa","meta","grid","qu"]
    out = set()
    for _ in range(80):
        stem = _ensure_len(_letters(base[:6] + random.choice(syll)))
        if len(stem) < 3:
            continue
        tld = random.choice(tlds)
        out.add(stem + tld)
        if len(out) >= 18:
            break
    # keep 12 unique
    return list(out)[:12]

def _mk_bad_domains(biz: str, tlds):
    # violates constraints: spaces, underscores, numbers, hyphens, too long
    base = re.sub(r"\s+", "-", biz.lower())
    out = set()
    for i in range(24):
        n = f"{base}-{i:02d}_{random.choice(['99','x','-'])}{random.choice(tlds)}"
        out.add(n)
        if len(out) >= 12:
            break
    return list(out)[:12]

def _lines_ok(lines, tlds):
    if not lines or len(lines) < 10:
        return False
    ok = 0
    for l in lines:
        if any(l.endswith(t) for t in tlds) and re.fullmatch(r"[a-z0-9\-\.]+", l):
            ok += 1
    return ok >= 10

def _positive_example(biz, tlds):
    sys_prompt = "You refuse unsafe requests and generate safe, brandable domains."
    user = PROMPT_TMPL.format(biz=biz, tlds=",".join(tlds))
    domains = _mk_good_domains(biz, tlds)
    return {
        "messages": [
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user},
            {"role":"assistant","content":"\n".join(domains)}
        ],
        "meta":{"type":"positive","business":biz,"tlds":tlds}
    }

def _refusal_example(unsafe_prompt):
    sys_prompt = "You must refuse inappropriate or harmful requests."
    user = f"Please generate domain ideas for: {unsafe_prompt}"
    return {
        "messages":[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user},
            {"role":"assistant","content": SAFE_REFUSAL}
        ],
        "meta":{"type":"refusal","unsafe":unsafe_prompt}
    }

def _negative_example(biz, tlds):
    # for contrastive evaluation/DPO later — NOT for SFT
    sys_prompt = "You refuse unsafe requests and generate safe, brandable domains."
    user = PROMPT_TMPL.format(biz=biz, tlds=",".join(tlds))
    bad = _mk_bad_domains(biz, tlds)
    return {
        "messages":[
            {"role":"system","content":sys_prompt},
            {"role":"user","content":user},
            {"role":"assistant","content":"\n".join(bad)}
        ],
        "meta":{"type":"negative","business":biz,"tlds":tlds}
    }

def _sanitize_lines(text: str):
    lines = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        s = s.lstrip("-*•1234567890. ").strip().lower()
        s = re.sub(r"[^a-z0-9\-.]", "", s)
        if s:
            lines.append(s)
        if len(lines) >= 30:
            break
    return lines

def _validate_and_fix(row, allowed_tlds):
    # Guarantee assistant content exists and is valid for positives
    t = row["meta"]["type"]
    if t == "positive":
        lines = _sanitize_lines(row["messages"][-1]["content"])
        lines = [l for l in lines if any(l.endswith(t) for t in allowed_tlds)]
        if not _lines_ok(lines, allowed_tlds):
            return None
        row["messages"][-1]["content"] = "\n".join(lines[:12])
    elif t == "refusal":
        # keep as-is
        pass
    elif t == "negative":
        # keep as-is for the negatives file
        pass
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/synthetic/seed_businesses.yaml")
    ap.add_argument("--safety", default="data/synthetic/safety_lists.yaml")
    ap.add_argument("--out_sft", required=True, help="JSONL with positive+refusal rows")
    ap.add_argument("--out_neg", required=True, help="JSONL with negative contrastives")
    ap.add_argument("--n", type=int, default=800, help="target positives (refusals added on top)")
    ap.add_argument("--tlds", default=".com,.io,.co,.ai,.app")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.seeds, "r", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)
    with open(args.safety, "r", encoding="utf-8") as f:
        saf = yaml.safe_load(f)

    allowed_tlds = [t.strip() for t in args.tlds.split(",") if t.strip().startswith(".")]

    # 1) Positives from seeds (diverse categories)
    categories = list(seeds.keys())
    positives = []
    negatives = []
    while len(positives) < args.n:
        cat = random.choice(categories)
        biz = random.choice(seeds[cat])
        pos = _positive_example(biz, allowed_tlds)
        neg = _negative_example(biz, allowed_tlds)
        pos = _validate_and_fix(pos, allowed_tlds)
        neg = _validate_and_fix(neg, allowed_tlds)
        if pos:
            positives.append(pos)
        if neg:
            negatives.append(neg)

    # 2) Refusals from banned keywords
    refusals = []
    for kw in saf.get("banned_keywords", []):
        refusals.append(_refusal_example(f"{kw} website"))
        refusals.append(_refusal_example(f"{kw} marketplace"))

    # 3) Write files
    sft_rows = positives + refusals
    neg_rows = negatives

    # Simple hash for reproducibility
    def _hash(rows):
        s = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    out_sft = Path(args.out_sft); out_sft.parent.mkdir(parents=True, exist_ok=True)
    out_neg = Path(args.out_neg); out_neg.parent.mkdir(parents=True, exist_ok=True)

    with open(out_sft, "w", encoding="utf-8") as w:
        for r in sft_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out_neg, "w", encoding="utf-8") as w:
        for r in neg_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "positives": len(positives),
        "refusals": len(refusals),
        "negatives": len(neg_rows),
        "tlds": allowed_tlds,
        "seed": args.seed,
        "sft_hash": _hash(sft_rows),
        "neg_hash": _hash(neg_rows),
    }
    with open(out_sft.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
