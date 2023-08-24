from typing import List
from mkr.retrievers.baseclass import Retriever, RetrieverOutput


class MultistageRetriever(Retriever):
    def __init__(self, sequence_retrievers: List[Retriever]):
        self.sequence_retrievers = sequence_retrievers

    def __call__(self, corpus_name: str, query: str, top_ks: List[int] = (100, 3)) -> RetrieverOutput:
        assert len(top_ks) == len(self.sequence_retrievers), "top_ks must have the same length as sequence_retrievers."

        candidate_ids = None
        for retriever in self.sequence_retrievers:
            results = retriever(corpus_name, query, top_k=top_ks[0], candidate_ids=candidate_ids)
            candidate_ids = [result["id"] for result in results]
        return results