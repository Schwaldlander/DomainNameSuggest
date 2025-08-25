from typing import List, Optional
try:
    from sentence_transformers import CrossEncoder
except Exception as e:
    CrossEncoder = None

# A light cross-encoder to score (business, domain) pairs for brandability/relevance.
# Uses ms-marco-MiniLM by default for speed. You can choose a stronger reranker later.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: Optional[str] = None):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers not installed correctly.")
        self.model = CrossEncoder(model_name, device=device)

    def score(self, business: str, domains: List[str]) -> List[float]:
        pairs = [[business, d] for d in domains]
        return self.model.predict(pairs).tolist()

    def rerank(self, business: str, domains: List[str]) -> List[str]:
        scores = self.score(business, domains)
        order = sorted(range(len(domains)), key=lambda i: scores[i], reverse=True)
        return [domains[i] for i in order]
