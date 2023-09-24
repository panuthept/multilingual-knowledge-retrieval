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
    corpus_dir = "./corpus/xquad"

    # load corpus
    corpus_index = json.load(open(f"{corpus_dir}/corpus_index.json", "r", encoding="utf-8"))

    # Create dataset directory
    if not os.path.exists("./datasets/xquad"):
        os.makedirs("./datasets/xquad")

    # Load training set to split into train, dev, and test set
    qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/qrel_test.jsonl", corpus_index)
    # Save
    with open("./datasets/xquad/qrel_test.jsonl", "w") as f:
        for qrel in qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")