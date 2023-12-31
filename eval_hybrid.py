import json
import argparse
from tqdm import tqdm
from mkr.retrievers.hybrid_retriever import HybridRetriever
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig
from mkr.retrievers.sparse_retriever import SparseRetriever, SparseRetrieverConfig


def eval(corpus_name, qrels, retrieval):
    # Initial metrics
    metrics = {
        "hit@1": 0,
        "hit@3": 0,
        "hit@5": 0,
        "hit@10": 0,
        "hit@30": 0,
        "hit@50": 0,
        "hit@100": 0,
        "mrr": 0,
    }
    # Evaluate
    sample_count = 0
    for question, target_document_ids in tqdm(qrels.items()):
        target_document_ids = set(target_document_ids)
        if len(target_document_ids) == 0:
            continue
        topk_results = retrieval(corpus_name, query=question, top_k=1000)
        retrieved_doc_ids = [result["id"] for result in topk_results]

        for rank, retrieved_doc_id in enumerate(retrieved_doc_ids):
            if retrieved_doc_id in target_document_ids:
                # Hit@k
                for k in [1, 3, 5, 10, 30, 50, 100]:
                    if rank < k:
                        metrics[f"hit@{k}"] += 1
                # MRR
                metrics["mrr"] += 1 / (rank + 1)
                break
        sample_count += 1
    # Normalize metrics
    for k in [1, 3, 5, 10, 30, 50, 100]:
        metrics[f"hit@{k}"] /= sample_count
    metrics["mrr"] /= sample_count
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_weight", type=float, default=0.5)
    args = parser.parse_args()

    # Prepare retriever
    dense_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name="mUSE",
            database_path="./database/mUSE",
        ),
    )
    dense_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    dense_retrieval.add_corpus("tydiqa_thai", "./corpus/tydiqa_thai/primary_corpus.jsonl")

    sparse_retrieval = SparseRetriever(
        SparseRetrieverConfig(
            database_path="./database/BM25",
        ),
    )
    sparse_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    sparse_retrieval.add_corpus("tydiqa_thai", "./corpus/tydiqa_thai/primary_corpus.jsonl")

    doc_retrieval = HybridRetriever(dense_retrieval, sparse_retrieval, sparse_weight=args.sparse_weight)

    # IAPP-WikiQA
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/iapp_wiki_qa/qrel_train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            qrels[question] = document_ids

    print("IAPP-WikiQA Performance:")
    eval_metrics = eval("iapp_wiki_qa", qrels, doc_retrieval)
    print(f"MRR@1000: {eval_metrics['mrr']:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        print(f"Hit@{k}: {eval_metrics[f'hit@{k}']:.4f}")
    ####################################################################################
    # TYDI-QA (Primary)
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/tydiqa_thai/primary_qrel_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            qrels[question] = document_ids

    print("TYDI-QA (Primary) Performance:")
    eval_metrics = eval("tydiqa_thai", qrels, doc_retrieval)
    print(f"MRR@1000: {eval_metrics['mrr']:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        print(f"Hit@{k}: {eval_metrics[f'hit@{k}']:.4f}")
    ####################################################################################