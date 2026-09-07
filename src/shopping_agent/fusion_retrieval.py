from __future__ import annotations

from .models import SearchHit, SearchRequest


class ScoreFusionRetriever:
    """Fuse lexical and dense scores over the same product catalog."""

    def __init__(self, lexical, semantic, lexical_weight: float = 0.65):
        if not 0 <= lexical_weight <= 1:
            raise ValueError("lexical_weight 必须在 0 到 1 之间")
        self.lexical = lexical
        self.semantic = semantic
        self.lexical_weight = lexical_weight
        self.products = semantic.products
        self.catalog_path = semantic.catalog_path

    def search(self, request: SearchRequest) -> list[SearchHit]:
        candidate_request = request.model_copy(update={"top_k": 20})
        semantic_hits = self.semantic.search(candidate_request)
        if not request.query:
            return semantic_hits[: request.top_k]
        lexical_hits = self.lexical.search(candidate_request)
        lexical_by_id = {hit.product.id: hit for hit in lexical_hits}
        semantic_by_id = {hit.product.id: hit for hit in semantic_hits}
        hits: list[SearchHit] = []
        for product_id in set(lexical_by_id) | set(semantic_by_id):
            lexical_hit = lexical_by_id.get(product_id)
            semantic_hit = semantic_by_id.get(product_id)
            product = (lexical_hit or semantic_hit).product
            lexical_score = lexical_hit.text_score if lexical_hit else 0.0
            semantic_score = semantic_hit.score if semantic_hit else 0.0
            score = self.lexical_weight * lexical_score + (1 - self.lexical_weight) * semantic_score
            reasons = []
            if lexical_score > 0:
                reasons.append("精确词项匹配")
            if semantic_hit:
                reasons.append("CLIP 图文语义相似")
            hits.append(SearchHit(product=product, score=round(score, 4), text_score=round(lexical_score, 4), image_score=round(semantic_score, 4), reasons=reasons))
        return sorted(hits, key=lambda hit: (-hit.score, -hit.product.rating, hit.product.id))[: request.top_k]
