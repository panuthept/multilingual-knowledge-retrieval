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
            if len(data["contexts"]) == 0:
                continue
            document_ids = list(data["contexts"])
            document_ids = [document_id for document_id in document_ids if document_id in corpus_index]
            if len(document_ids) == 0:
                continue
            qrels.append({"question": question, "document_ids": document_ids})
    return qrels


if __name__ == "__main__":
    input_dir = "./corpus/miracl"
    corpus_dir = "./corpus/miracl"

    # load corpus
    corpus_index = json.load(open(f"{corpus_dir}/corpus_index.json", "r", encoding="utf-8"))

    # Create dataset directory
    if not os.path.exists("./datasets/miracl"):
        os.makedirs("./datasets/miracl")

    # Load training set to split into train, dev, and test set
    qrels: List[Dict[str, str]] = load_qrels(f"{input_dir}/qrel_train.jsonl", corpus_index)
    qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/qrel_dev.jsonl", corpus_index)
    # Split into train and dev set
    qrels_train = qrels[:int(len(qrels) * 0.9)]
    qrels_dev = qrels[int(len(qrels) * 0.9):]
    # Save
    with open("./datasets/miracl/qrel_train.jsonl", "w") as f:
        for qrel in qrels_train:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open("./datasets/miracl/qrel_dev.jsonl", "w") as f:
        for qrel in qrels_dev:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open("./datasets/miracl/qrel_test.jsonl", "w") as f:
        for qrel in qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")