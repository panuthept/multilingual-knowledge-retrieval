import json
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig


def eval_hit(corpus_name, qrels, retrieval, top_k=1):
    output = retrieval(corpus_name, queries=list(qrels.keys()), top_k=top_k)

    hit_at_k = 0
    for question, topk_results in zip(output.queries, output.results):
        target_document_ids = set(qrels[question])
        retrieved_doc_ids = [result["id"] for result in topk_results]
        for retrieved_doc_id in retrieved_doc_ids:
            if retrieved_doc_id in target_document_ids:
                hit_at_k += 1
                break
    hit_at_k = hit_at_k / len(qrels)
    return hit_at_k


def eval_mrr(corpus_name, qrels, retrieval, top_k=1000):
    output = retrieval(corpus_name, queries=list(qrels.keys()), top_k=top_k)

    mrrs = []
    for question, topk_results in zip(output.queries, output.results):
        target_document_ids = set(qrels[question])
        retrieved_doc_ids = [result["id"] for result in topk_results]

        mrr = 0
        for rank, retrieved_doc_id in enumerate(retrieved_doc_ids):
            if retrieved_doc_id in target_document_ids:
                mrr = 1 / (rank + 1)
                break
        mrrs.append(mrr)
    avg_mrr = sum(mrrs) / len(mrrs)
    return avg_mrr


if __name__ == "__main__":
    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name="mUSE",
            database_path="./database/mUSE",
        ),
    )
    doc_retrieval.add_corpus("iapp_wiki_qa", "./corpus/iapp_wiki_qa/corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa_primary", "./corpus/tydiqa/primary_corpus.jsonl")
    doc_retrieval.add_corpus("tydiqa_secondary", "./corpus/tydiqa/secondary_corpus.jsonl")

    # IAPP-WikiQA
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/iapp_wiki_qa/qrels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            qrels[question] = document_ids

    print("IAPP-WikiQA Performance:")
    mrr = eval_mrr("iapp_wiki_qa", qrels, doc_retrieval, top_k=1000)
    print(f"MRR@1000: {mrr:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval_hit("iapp_wiki_qa", qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k:.4f}")
    ####################################################################################
    # TYDI-QA (Primary)
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/tydiqa/primary_qrel_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            qrels[question] = document_ids

    print("TYDI-QA (Primary) Performance:")
    mrr = eval_mrr("tydiqa_primary", qrels, doc_retrieval, top_k=1000)
    print(f"MRR@1000: {mrr:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval_hit("tydiqa_primary", qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k:.4f}")
    ####################################################################################
    # TYDI-QA (Secondary)
    ####################################################################################
    # Load qrels
    qrels = {}  # {question: document_ids}
    with open("./corpus/tydiqa/secondary_qrel_val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            document_ids = data["context_answers"].keys()
            if question == "":
                continue
            qrels[question] = document_ids

    print("TYDI-QA (Secondary) Performance:")
    mrr = eval_mrr("tydiqa_secondary", qrels, doc_retrieval, top_k=1000)
    print(f"MRR@1000: {mrr:.4f}")
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval_hit("tydiqa_secondary", qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k:.4f}")
    ####################################################################################