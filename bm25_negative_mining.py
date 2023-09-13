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
        with open("./corpus/training_dataset/bm25_top1000.jsonl", "a", encoding="utf-8") as f:
            f.write(f"{json.dumps({'question': question, 'top1000': retrieved_doc_ids}, ensure_ascii=False)}\n")

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
    # Prepare retriever
    doc_retrieval = SparseRetriever(
        SparseRetrieverConfig(
            database_path="./database/BM25",
        ),
    )
    doc_retrieval.add_corpus("training_dataset", "./corpus/training_dataset/corpus.jsonl")

    top1000 = {}
    with open("./corpus/training_dataset/bm25_top1000.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            top1000[question] = data["top1000"]

    # Training set
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/training_dataset/qrel_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            if question in top1000:
                continue
            qrels[question] = document_ids

    print("Training set Performance:")
    eval_metrics = eval("training_dataset", qrels, doc_retrieval)
    print(f"MRR@1000: {eval_metrics['mrr']:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        print(f"Hit@{k}: {eval_metrics[f'hit@{k}']:.4f}")
    ####################################################################################