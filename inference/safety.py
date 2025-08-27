import json, regex as re, os

DEFAULT_DECODING = {
    "temperature": 0.45,
    "top_p": 0.85,
    "repetition_penalty": 1.1,
    "max_new_tokens": 192,
    "do_sample": True,
}

SAFE_REFUSAL = "I can’t help with that request, but I can suggest safe, appropriate domain ideas for permitted businesses."

BANNED = [
    "adult","porn","cocaine","heroin","meth","gun","weapon","hitman",
    "hate","terror","nazis","suicide","self-harm","anorexia","bulimia",
    "fakeid","counterfeit","steroids","casino","gambling","sextape"
]

BAD_PATTERNS = [
    r"(?i)adult|porn|cocaine|heroin|meth|hitman|terror|nazis|suicide|self-?harm|anorexia|bulimia|fakeid|counterfeit|steroids|casino|gambling|sextape",
]

def safe_decode_config():
    # Allow overriding via configs/decoding.json
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "decoding.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {**DEFAULT_DECODING, **d}
    return DEFAULT_DECODING

def needs_refusal(prompt: str) -> bool:
    p = prompt.lower()
    return any(b in p for b in BANNED)

def sanitize_output(text: str) -> str:
    # Remove extra commentary; keep up to 30 lines, alnum-dash-dot only
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # strip bullets
        s = s.lstrip("-*•1234567890. ").strip()
        s = re.sub(r"[^a-z0-9\-.]", "", s.lower())
        if s:
            lines.append(s)
        if len(lines) >= 30:
            break
    return "\n".join(lines)

def filter_unsafe(lines):
    pat = re.compile("|".join(BAD_PATTERNS))
    return [l for l in lines if not pat.search(l)]
