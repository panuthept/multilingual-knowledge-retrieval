import json
from typing import List


def read_corpus(corpus_dir: str):
    corpus: List[str] = []
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data["doc_text"])
    return corpus