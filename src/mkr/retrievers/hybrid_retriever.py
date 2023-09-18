from typing import List, Dict, Any
from mkr.retrievers.baseclass import Retriever
from mkr.utilities.general_utils import normalize_score


class HybridRetriever(Retriever):
    def __init__(self, dense_retriever: Retriever, sparse_retriever: Retriever, sparse_weight: float = 0.5):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.sparse_weight = sparse_weight

    def __call__(
            self, 
            corpus_name: str,
            query: str, 
            top_k: int = 3, 
            candidate_ids: List[str] = None, 
        ) -> List[Dict[str, Any]]:
        dense_results: List[Dict[str, Any]] = self.dense_retriever(corpus_name, query, top_k=top_k, candidate_ids=candidate_ids)
        sparse_results: List[Dict[str, Any]] = self.sparse_retriever(corpus_name, query, top_k=top_k, candidate_ids=candidate_ids)

        combined_results = {}
        for dense_result in dense_results:
            combined_results[dense_result["id"]] = {
                "id": dense_result["id"],
                "score": dense_result["score"] * (1 - self.sparse_weight),
                "content": dense_result["content"],
                "metadata": dense_result["metadata"],
            }
        for sparse_result in sparse_results:
            if sparse_result["id"] not in combined_results:
                combined_results[sparse_result["id"]] = {
                    "id": sparse_result["id"],
                    "score": sparse_result["score"] * self.sparse_weight,
                    "content": sparse_result["content"],
                    "metadata": sparse_result["metadata"],
                }
            else:
                combined_results[sparse_result["id"]]["score"] += sparse_result["score"] * self.sparse_weight
        # Convert combined_results to list
        combined_results = list(combined_results.values())
        # Sort by score
        combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)
        # Normalize score
        combined_results = normalize_score(combined_results)
        return combined_results