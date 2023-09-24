import os
import math
import json
from tqdm import tqdm
from typing import Dict, List
from dataclasses import dataclass, field
from mkr.retrievers.baseclass import Retriever
from mkr.resources.resource_manager import ResourceManager


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
    def __init__(self, resource_management: ResourceManager, retriever: Retriever):
        self.resource_management = resource_management
        self.retriever = retriever

    def get_qrels(self, dataset_path: str):
        # Load qrels
        qrels = []
        with open(os.path.join(dataset_path, "qrel_test.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                qrels.append(json.loads(line))
        return qrels

    def evaluate_on_dataset(self, corpus_name: str, qrels: List[Dict[str, List[str]]]) -> Dict[str, float]:
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
    
    def evaluate_on_datasets(self, dataset_names: List[str]):
        for dataset_name in dataset_names:
            # Get dataset path
            dataset_path = self.resource_management.get_dataset_path(dataset_name)
            # Get qrels
            qrels = self.get_qrels(dataset_path)
            # Evaluate
            results = self.evaluate_on_dataset(dataset_name, qrels)
            # Report results
            print("*" * 50)
            print(f"Dataset: {dataset_name.upper()}")
            for key, values in results.items():
                print(f"{key}: {values * 100:.1f}")
            print("*" * 50)