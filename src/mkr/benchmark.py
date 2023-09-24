import math
from tqdm import tqdm
from typing import Dict, List
from dataclasses import dataclass, field
from mkr.retrievers.baseclass import Retriever


@dataclass
class Metrics:
    hit_indices: List[int] = field(default_factory=list)

    def __add__(self, other: 'Metrics'):
        return Metrics(
            hit_indices=self.hit_indices + other.hit_indices,
        )

    def get_mrr(self, k: int = 1000):
        assert len(self.hit_indices) > 0, "No hit indices"
        mrr = [1.0 / (hti_index + 1) if hti_index <= k else 0.0 for hti_index in self.hit_indices]
        return sum(mrr) / len(mrr)

    def get_recall(self, k: int = 1000):
        assert len(self.hit_indices) > 0, "No hit indices"
        recall = [float(hti_index <= k) for hti_index in self.hit_indices]
        return sum(recall) / len(recall)


class Benchmark:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def evaluate(self, corpus_name: str, qrels: List[Dict[str, List[str]]]) -> Dict[str, float]:
        # Intiial metrics
        metrics = Metrics()
        for qrel in tqdm(qrels, desc="Evaluating"):
            question = qrel["question"]
            gold_document_ids = set(qrel["document_ids"])

            topk_results = self.retriever(corpus_name, query=question, top_k=1000)
            retrieved_doc_ids = [result["id"] for result in topk_results]

            is_hit = False
            for rank, retrieved_doc_id in enumerate(retrieved_doc_ids):
                if retrieved_doc_id in gold_document_ids:
                    metrics.hit_indices.append(rank)
                    is_hit = True
                    break
            if not is_hit:
                metrics.hit_indices.append(math.inf)
        # Get metrics
        return {
            "MRR": metrics.get_mrr(k=1000),
            "R@5": metrics.get_recall(k=5),
            "R@1000": metrics.get_recall(k=1000),
        }