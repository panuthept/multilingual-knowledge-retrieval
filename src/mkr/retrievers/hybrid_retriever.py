from typing import List, Dict
from mkr.utilities.general_utils import normalize_score
from mkr.retrievers.baseclass import Retriever, RetrieverOutput


class HybridRetriever(Retriever):
    def __init__(self, dense_retriever: Retriever, sparse_retriever: Retriever):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    def __call__(self, queries: List[str], top_k: int = 3, sparse_weight: float = 0.5) -> RetrieverOutput:
        combined_resultss = []
        dense_resultss: List[List[Dict]] = self.dense_retriever(queries, top_k=top_k).resultss
        sparse_resultss: List[List[Dict]] = self.sparse_retriever(queries, top_k=top_k).resultss
        for dense_results, sparse_results in zip(dense_resultss, sparse_resultss):
            combined_results = {}
            for dense_result in dense_results.values():
                combined_results[dense_result["doc_id"]] = {
                    "doc_id": dense_result["doc_id"],
                    "score": dense_result["score"] * (1 - sparse_weight),
                    "doc_url": dense_result["doc_url"],
                    "doc_title": dense_result["doc_title"],
                    "doc_text": dense_result["doc_text"],
                }
            for sparse_result in sparse_results.values():
                if sparse_result["doc_id"] not in combined_results:
                    combined_results[sparse_result["doc_id"]] = {
                        "doc_id": sparse_result["doc_id"],
                        "score": sparse_result["score"] * sparse_weight,
                        "doc_url": sparse_result["doc_url"],
                        "doc_title": sparse_result["doc_title"],
                        "doc_text": sparse_result["doc_text"],
                    }
                else:
                    combined_results[sparse_result["doc_id"]]["score"] += sparse_result["score"] * sparse_weight
            combined_results = {doc_result["doc_id"]: doc_result for doc_result in sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)[:top_k]}
            combined_resultss.append(combined_results)
        # Normalize score
        combined_resultss = normalize_score(combined_resultss)
        return RetrieverOutput(
            queries=queries,
            resultss=combined_resultss,
        )