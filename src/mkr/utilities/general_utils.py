import json
from typing import List, Dict, Any


def read_corpus(corpus_dir: str):
    corpus: List[Dict[str, str]] = []
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data)
    return corpus

def normalize_score(results: List[Dict[str, Any]]):
    max_score = max([result["score"] for result in results])
    for result in results:
        result["score"] = result["score"] / max_score
    return results