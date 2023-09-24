import json
from tqdm import tqdm
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
        "hit@1000": 0,
        "mrr@1000": 0,
    }
    # Evaluate
    sample_count = 0
    for sample in tqdm(qrels):
        question = sample["question"]
        target_document_ids = sample["document_ids"]
        target_document_ids = set(target_document_ids)
        if len(target_document_ids) == 0:
            continue
        topk_results = retrieval(corpus_name, query=question, top_k=1000)
        retrieved_doc_ids = [result["id"] for result in topk_results]

        for rank, retrieved_doc_id in enumerate(retrieved_doc_ids):
            if retrieved_doc_id in target_document_ids:
                # Hit@k
                for k in [1, 3, 5, 10, 30, 50, 100, 1000]:
                    if rank < k:
                        metrics[f"hit@{k}"] += 1
                # MRR
                metrics["mrr@1000"] += 1 / (rank + 1)
                break
        sample_count += 1
    # Normalize metrics
    for k in [1, 3, 5, 10, 30, 50, 100, 1000]:
        metrics[f"hit@{k}"] /= sample_count
    metrics["mrr@1000"] /= sample_count
    return metrics


if __name__ == "__main__":
    # Prepare retriever
    doc_retrieval = SparseRetriever(
        SparseRetrieverConfig(
            database_path="./database/BM25",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa", "./corpus/tydiqa/corpus.jsonl")

    # IAPP-WikiQA
    ####################################################################################
    # Load qrels
    qrels = []
    with open("./datasets/iapp_wiki_qa/qrel_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    print("IAPP-WikiQA Performance:")
    eval_metrics = eval("iapp_wiki_qa", qrels, doc_retrieval)
    print(f"MRR@1000: {eval_metrics['mrr@1000'] * 100:.1f}")
    for k in [1, 3, 5, 10, 30, 50, 100, 1000]:
        print(f"Hit@{k}: {eval_metrics[f'hit@{k}'] * 100:.1f}")
    ####################################################################################
    # TYDI-QA (Primary)
    ####################################################################################
    # Load qrels
    qrels = []
    with open("./datasets/tydiqa/qrel_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qrels.append(json.loads(line))

    print("TYDI-QA (Primary) Performance:")
    eval_metrics = eval("tydiqa", qrels, doc_retrieval)
    print(f"MRR@1000: {eval_metrics['mrr@1000'] * 100:.1f}")
    for k in [1, 3, 5, 10, 30, 50, 100, 1000]:
        print(f"Hit@{k}: {eval_metrics[f'hit@{k}'] * 100:.1f}")
    ####################################################################################