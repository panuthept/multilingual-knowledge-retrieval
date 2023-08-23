import json
from mkr.retrievers.dense_retriever import DenseRetriever, DenseRetrieverConfig


def eval(qrels, retrieval, top_k=1):
    output = retrieval(queries=list(qrels.keys()), top_k=top_k)

    hit_at_k = 0
    for question, topk_results in zip(output.queries, output.results):
        target_document_ids = qrels[question]
        retrieved_doc_ids = set([result["id"] for result in topk_results])
        is_hit = False
        for target_document_id in target_document_ids:
            if target_document_id in retrieved_doc_ids:
                is_hit = True
                break
        hit_at_k += int(is_hit)
    hit_at_k = hit_at_k / len(qrels)
    return hit_at_k


if __name__ == "__main__":
    # IAPP-WikiQA
    ####################################################################################
    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name="mUSE",
            database_path="./database/mUSE/iapp_wiki_qa",
        ),
    )
    # doc_retrieval.add_corpus("./corpus/iapp_wiki_qa/corpus.jsonl")

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
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval(qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k * 100:.2f}%")
    ####################################################################################
    # TYDI-QA (Primary)
    ####################################################################################
    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name="mUSE",
            database_path="./database/mUSE/tydiqa_primary",
        ),
    )
    doc_retrieval.add_corpus("./corpus/tydiqa/primary_corpus.jsonl")

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
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval(qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k * 100:.2f}%")
    ####################################################################################
    # TYDI-QA (Secondary)
    ####################################################################################
    # Prepare retriever
    doc_retrieval = DenseRetriever(
        DenseRetrieverConfig(
            model_name="mUSE",
            database_path="./database/mUSE/tydiqa_secondary",
        ),
    )
    doc_retrieval.add_corpus("./corpus/tydiqa/secondary_corpus.jsonl")

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
    for k in [1, 3, 5, 10, 30, 50, 100]:
        hit_at_k = eval(qrels, doc_retrieval, top_k=k)
        print(f"Hit@{k}: {hit_at_k * 100:.2f}%")
    ####################################################################################