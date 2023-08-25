from typing import List, Dict, Any
from mkr.retrievers.baseclass import Retriever


class MultistageRetriever(Retriever):
    def __init__(self, sequence_retrievers: List[Retriever]):
        self.sequence_retrievers = sequence_retrievers

    def __call__(
            self, 
            corpus_name: str, 
            query: str, 
            top_ks: List[int] = (100, 3), 
            candidate_ids: List[str] = None
        ) -> List[Dict[str, Any]]:
        assert len(top_ks) == len(self.sequence_retrievers), "top_ks must have the same length as sequence_retrievers."

        for retriever in self.sequence_retrievers:
            results = retriever(corpus_name, query, top_k=top_ks[0], candidate_ids=candidate_ids)
            candidate_ids = [result["id"] for result in results]
        return results