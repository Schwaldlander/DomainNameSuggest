# inference/judge.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json, regex as re, math, torch
from .loader import load_eval_model

GENERIC_STOPWORDS = set("""
best top the my your our shop store online official app site hub world pro
services solution solutions corp company inc ltd global central network plus
""".split())

JUDGE_SYSTEM = (
    "You are a strict branding evaluator. "
    "Given a business context and candidate domain names, you will return a JSON array with an item for each domain:\n"
    '{ "domain": "name.tld", "score": 0..1, "generic": true|false, "reason": "short human-readable reason" }.\n'
    "Scoring rubric (0..1):\n"
    "- 0.9–1.0: Distinctive, highly brandable, short, memorable.\n"
    "- 0.7–0.89: Strong; fits business; not generic; pronounceable.\n"
    "- 0.5–0.69: Acceptable but somewhat bland or derivative.\n"
    "- <0.5: Unappealing, generic, spammy, too long, or violates constraints.\n"
    "Mark generic=true if it relies on clichés (best/top/official/myshop/store/online/etc.) or is overly literal.\n"
    "Keep answers concise."
)

JUDGE_USER_TMPL = """Business: {business}
Allowed TLDs: {tlds}
Evaluate these domains strictly and return a pure JSON array (no code fences). One item per domain:
{lines}
"""

def _batched(items: List[str], bsz: int) -> List[List[str]]:
    return [items[i:i+bsz] for i in range(0, len(items), bsz)]

def _heuristic_generic(name: str) -> bool:
    # simple heuristic fallback on SLD (pre-TLD)
    sld = name.split(".", 1)[0]
    tokens = re.findall(r"[a-z]+", sld.lower())
    return any(tok in GENERIC_STOPWORDS for tok in tokens)

def _heuristic_score(name: str) -> float:
    sld = name.split(".", 1)[0]
    n = len(sld)
    base = 0.6
    # sweet spot length 5..10
    base += max(0.0, 0.25 - abs(7 - n) * 0.05)
    if re.search(r"[\d_ ]", sld) or "-" in sld:
        base -= 0.2
    if _heuristic_generic(name):
        base -= 0.25
    return float(max(0.0, min(1.0, base)))

def _safe_json_parse(text: str) -> List[Dict[str, Any]] | None:
    text = text.strip()
    # try to locate first json array
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    js = m.group(0)
    try:
        return json.loads(js)
    except Exception:
        js = js.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("`", "'")
        js = re.sub(r",\s*([}\]])", r"\1", js)
        return json.loads(js)

def judge_domains(
    model,
    tokenizer,
    business: str,
    domains: List[str],
    allowed_tlds: List[str],
    batch_size: int = 25,
    device: str | None = None
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: {"domain": str, "score": float, "generic": bool, "reason": str}
    Falls back to heuristic if LLM JSON parsing fails for a batch.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    results: List[Dict[str, Any]] = []
    for chunk in _batched(domains, batch_size):
        lines = "\n".join(f"- {d}" for d in chunk)
        user = JUDGE_USER_TMPL.format(business=business, tlds=", ".join(allowed_tlds), lines=lines)
        ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                max_new_tokens=512,
                temperature=0.2,     # deterministic-ish judger
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.05,
            )
        text = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        parsed = _safe_json_parse(text)
        if isinstance(parsed, list) and parsed:
            # normalize items
            found_by_domain = { (item.get("domain") or "").lower(): item for item in parsed if isinstance(item, dict) }
            for d in chunk:
                item = found_by_domain.get(d.lower())
                if isinstance(item, dict):
                    score = float(item.get("score", _heuristic_score(d)))
                    generic = bool(item.get("generic", _heuristic_generic(d)))
                    reason = str(item.get("reason", "")).strip() or "—"
                else:
                    score = _heuristic_score(d)
                    generic = _heuristic_generic(d)
                    reason = "heuristic fallback"
                results.append({"domain": d, "score": score, "generic": generic, "reason": reason})
        else:
            # fallback for entire chunk
            for d in chunk:
                results.append({"domain": d, "score": _heuristic_score(d), "generic": _heuristic_generic(d), "reason": "heuristic fallback"})
    return results



if __name__ == "__main__":
    import argparse, json, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--business", required=True)
    ap.add_argument("--base_model",required=True)
    ap.add_argument("--domains", required=True, help="Comma-separated domains")
    ap.add_argument("--tlds", default=".com,.io,.co,.ai,.app")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=25)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok, model = load_eval_model(
    model_path=args.model,
    base_model=args.base_model,
    device=device or ("auto"),
    load_4bit=True,
    local_only=True,
    )
    mdl = model.to(device) if device in ("cpu", "cuda") else model
    mdl.eval()

    print("Loading Completed")
    doms = [d.strip() for d in args.domains.split(",") if d.strip()]
    tlds = [t.strip() for t in args.tlds.split(",") if t.strip().startswith(".")]
    scored = judge_domains(mdl, tok, args.business, doms, tlds, batch_size=args.batch_size, device=device)
    print(json.dumps(scored, indent=2, ensure_ascii=False))
