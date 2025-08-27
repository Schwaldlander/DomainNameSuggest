import regex as re

SAFE_REFUSAL = "I can’t help with that request, but I can suggest safe, appropriate domain ideas for permitted businesses."

BANNED = [
    "adult","porn","cocaine","heroin","meth","gun","weapon","hitman","hate","terror","nazis",
    "suicide","self-harm","anorexia","bulimia","fakeid","counterfeit","steroids","casino","gambling","sextape"
]

REFUSAL_PAT = re.compile(r"(?i)(can[’']?t help|cannot assist|not able to help|violates policy)")

def should_refuse(prompt: str) -> bool:
    return any(b in prompt.lower() for b in BANNED)

def is_refusal(text: str) -> bool:
    return bool(REFUSAL_PAT.search(text))
