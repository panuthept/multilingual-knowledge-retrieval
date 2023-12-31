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
    input_dir = "./corpus/xquad"
    output_dir = "./datasets/question_answering/xquad"

    # load corpus
    th_corpus_index = json.load(open(f"{input_dir}/th_corpus_index.json", "r", encoding="utf-8"))
    en_corpus_index = json.load(open(f"{input_dir}/en_corpus_index.json", "r", encoding="utf-8"))

    # Create dataset directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load test sets
    th_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/th_qrel_test.jsonl", th_corpus_index)
    en_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/en_qrel_test.jsonl", en_corpus_index)
    th_en_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/th_en_qrel_test.jsonl", en_corpus_index)
    en_th_qrels_test: List[Dict[str, str]] = load_qrels(f"{input_dir}/en_th_qrel_test.jsonl", th_corpus_index)

    # Save
    with open(f"{output_dir}/th_qrel_test.jsonl", "w") as f:
        for qrel in th_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/en_qrel_test.jsonl", "w") as f:
        for qrel in en_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/th_en_qrel_test.jsonl", "w") as f:
        for qrel in th_en_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")
    with open(f"{output_dir}/en_th_qrel_test.jsonl", "w") as f:
        for qrel in en_th_qrels_test:
            f.write(f"{json.dumps(qrel, ensure_ascii=False)}\n")