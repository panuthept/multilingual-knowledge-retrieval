from typing import List, Dict, Any
from mkr.retrievers.baseclass import Retriever


class TwoStageRetriever(Retriever):
    def __init__(self, retriever: Retriever, reranker: Retriever):
        self.retriever = retriever
        self.reranker = reranker

    def __call__(
            self, 
            corpus_name: str, 
            query: str, 
            top_ks: List[int] = (100, 3), 
            candidate_ids: List[str] = None
        ) -> List[Dict[str, Any]]:
        results = self.retriever(corpus_name, query, top_k=top_ks[0], candidate_ids=candidate_ids)
        candidate_ids = [result["id"] for result in results]
        results = self.reranker(corpus_name, query, candidate_ids=candidate_ids, top_k=top_ks[1])
        return results