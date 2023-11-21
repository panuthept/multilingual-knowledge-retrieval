import os
import json
from typing import Dict, List


def load_qrels(file_path: str, corpus_index: Dict[str, str]):
    qrels: List[Dict[str, str]] = []  # [{question: document_ids}, ...]
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            if question == "":
                continue
            if len(data["context_answers"]) == 0:
                continue
            document_ids = list(data["context_answers"].keys())
            document_ids = [document_id for document_id in document_ids if document_id in corpus_index]
            if len(document_ids) == 0:
                continue
            qrels.append({"question": question, "document_ids": document_ids})
    return qrels


if __name__ == "__main__":
    input_dir = "./corpus/xquad"
    output_dir = "./datasets/retrieval"

    # load corpus
    th_corpus_index = json.load(open(f"{input_dir}/th_corpus_index.json", "r", encoding="utf-8"))
    en_corpus_index = json.load(open(f"{input_dir}/en_corpus_index.json", "r", encoding="utf-8"))

    # Create dataset directory
    if not os.path.exists(os.path.join(output_dir, "th_xquad")):
        os.makedirs(os.path.join(output_dir, "th_xquad"))
    if not os.path.exists(os.path.join(output_dir, "en_xquad")):
        os.makedirs(os.path.join(output_dir, "en_xquad"))
    if not os.path.exists(os.path.join(output_dir, "th_en_xquad")):
        os.makedirs(os.path.join(output_dir, "th_en_xquad"))
    if not os.path.exists(os.path.join(output_dir, "en_th_xquad")):
        os.makedirs(os.path.join(output_dir, "en_th_xquad"))

    # Load test sets
    th_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/th_qrel_test.jsonl", th_corpus_index)
    en_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/en_qrel_test.jsonl", en_corpus_index)
    th_en_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/th_en_qrel_test.jsonl", en_corpus_index)
    en_th_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/en_th_qrel_test.jsonl", th_corpus_index)

    # Save
    with open(f"{output_dir}/th_xquad/qrel_test.jsonl", "w") as f:
        for qrel in th_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/en_xquad/qrel_test.jsonl", "w") as f:
        for qrel in en_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/th_en_xquad/qrel_test.jsonl", "w") as f:
        for qrel in th_en_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/en_th_xquad/qrel_test.jsonl", "w") as f:
        for qrel in en_th_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")