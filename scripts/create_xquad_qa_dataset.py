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
            if len(data["context_answers"]) != 1:
                continue
            context_id = list(data["context_answers"].keys())[0]
            if context_id not in corpus_index:
                continue
            if len(data["context_answers"][context_id]) != 1:
                continue
            answer = data["context_answers"][context_id][0]
            if answer[0] == "":
                continue
            qrels.append({"question": question, "context_id": context_id, "answer": answer})
    return qrels


if __name__ == "__main__":
    input_dir = "./corpus/xquad_original"
    corpus_dir = "./corpus/xquad_original"
    output_dir = "./datasets/question_answering/xquad"

    # load corpus
    corpus_index = json.load(open(f"{corpus_dir}/corpus_index.json", "r", encoding="utf-8"))

    # Create dataset directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load training set to split into train, dev, and test set
    qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/qrel_test.jsonl", corpus_index)
    # Save
    with open(f"{output_dir}/qrel_test.jsonl", "w") as f:
        for qrel in qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")