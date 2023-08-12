import json
from typing import List, Dict


def read_corpus(corpus_dir: str):
    corpus: List[Dict[str, str]] = []
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data)
    return corpus

def normalize_score(resultss: List[List[Dict]]):
    for results in resultss:
        sum_score = sum([result["score"] for result in results]) + 1e-7
        for result in results:
            result["score"] /= sum_score
    return resultss