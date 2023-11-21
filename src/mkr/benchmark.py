import os
import math
import json
import Levenshtein
from tqdm import tqdm
from typing import Dict, List
from dataclasses import dataclass, field
from mkr.retrievers.baseclass import Retriever
from mkr.resources.resource_manager import ResourceManager
from mkr.question_answering.answer_extractor import AnswerExtractor


@dataclass
class RetrievalMetrics:
    hit_indices: List[int] = field(default_factory=list)

    def __add__(self, other: 'RetrievalMetrics'):
        return RetrievalMetrics(
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


class RetrievalBenchmark:
    def __init__(self, resource_management: ResourceManager, retriever: Retriever):
        self.resource_management = resource_management
        self.retriever = retriever

    def _get_dataset_path(self, dataset_name: str):
        return self.resource_management.get_dataset_path(dataset_name, dataset_type="retrieval")

    def get_qrels(self, dataset_path: str):
        # Load qrels
        qrels = []
        with open(os.path.join(dataset_path, "qrel_test.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                qrels.append(json.loads(line))
        return qrels

    def evaluate_on_dataset(self, corpus_name: str, qrels: List[Dict[str, List[str]]]) -> Dict[str, float]:
        # Intiial metrics
        metrics = RetrievalMetrics()
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
            "R@1": metrics.get_recall(k=1),
            "R@5": metrics.get_recall(k=5),
            "R@1000": metrics.get_recall(k=1000),
        }
    
    def evaluate_on_datasets(self, dataset_names: List[str], corpus_names: List[str]):
        for dataset_name, corpus_name in zip(dataset_names, corpus_names):
            # Get dataset path
            dataset_path = self._get_dataset_path(dataset_name)
            # Get qrels
            qrels = self.get_qrels(dataset_path)
            # Evaluate
            results = self.evaluate_on_dataset(corpus_name, qrels)
            # Report results
            print("*" * 50)
            print(f"Dataset: {dataset_name.upper()}")
            for key, values in results.items():
                print(f"{key}: {values * 100:.1f}")
            print("*" * 50)


@dataclass
class QAMetrics:
    preds: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def __add__(self, other: 'QAMetrics'):
        return QAMetrics(
            preds=self.preds + other.preds,
            labels=self.labels + other.labels,
        )
    
    def get_em_score(self):
        exact_matchs = []
        for pred, label in zip(self.preds, self.labels):
            exact_matchs.append(int(pred == label))
        return sum(exact_matchs) / len(exact_matchs)
    
    def get_string_similarity(self):
        similarities = []
        for pred, label in zip(self.preds, self.labels):
            similarities.append(Levenshtein.ratio(pred, label))
        return sum(similarities) / len(similarities)


class ClosedBookQABenchmark(RetrievalBenchmark):
    def __init__(self, resource_management: ResourceManager, extractor: AnswerExtractor):
        self.resource_management = resource_management
        self.extractor = extractor

    def _get_dataset_path(self, dataset_name: str):
        return self.resource_management.get_dataset_path(dataset_name, dataset_type="question_answering")

    def _load_corpus(self, corpus_name: str) -> Dict[str, str]:
        corpus_path = self.resource_management.get_corpus_path(corpus_name)
        corpus: Dict[str, str] = {}
        with open(os.path.join(corpus_path, "corpus.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc["hash"]] = doc["content"]
        return corpus

    def evaluate_on_dataset(self, corpus_name: str, qrels: List[Dict[str, List[str]]]) -> Dict[str, float]:
        corpus: Dict[str, str] = self._load_corpus(corpus_name)
        
        # Intiial metrics
        metrics = QAMetrics()
        for qrel in tqdm(qrels, desc="Evaluating"):
            question = qrel["question"]
            context = corpus[qrel["context_id"]]
            gold_answer = qrel["answer"][0]

            pred_answer = self.extractor(question, context)

            metrics.preds.append(pred_answer)
            metrics.labels.append(gold_answer)
        # Get metrics
        return {
            "EM": metrics.get_em_score(),
            "String Similarity": metrics.get_string_similarity(),
        }
    

class OpenedBookQABenchmark(ClosedBookQABenchmark):
    def __init__(self, resource_management: ResourceManager, retriever: Retriever, extractor: AnswerExtractor):
        self.resource_management = resource_management
        self.retriever = retriever
        self.extractor = extractor

    def evaluate_on_dataset(self, corpus_name: str, qrels: List[Dict[str, List[str]]]) -> Dict[str, float]:
        corpus: Dict[str, str] = self._load_corpus(corpus_name)
        
        # Intiial metrics
        metrics = QAMetrics()
        for qrel in tqdm(qrels, desc="Evaluating"):
            question = qrel["question"]
            gold_answer = qrel["answer"][0]

            topk_results = self.retriever(corpus_name, query=question, top_k=5)
            retrieved_doc_ids = [result["id"] for result in topk_results]

            pred_context = "\n".join([corpus[retrieved_doc_id] for retrieved_doc_id in retrieved_doc_ids])
            pred_answer = self.extractor(question, pred_context)

            metrics.preds.append(pred_answer)
            metrics.labels.append(gold_answer)
        # Get metrics
        return {
            "EM": metrics.get_em_score(),
            "String Similarity": metrics.get_string_similarity(),
        }