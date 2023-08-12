import json
from typing import List, Dict, Any


def read_corpus(corpus_dir: str):
    corpus: List[Dict[str, str]] = []
    with open(corpus_dir, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            corpus.append(data)
    return corpus

def normalize_score(lst_results: List[Dict[str, Dict[str, Any]]]):
    for results in lst_results:
        sum_score = sum([results[doc_id]["score"] for doc_id in results.keys()]) + 1e-7
        for doc_id in results.keys():
            results[doc_id]["score"] /= sum_score
    return lst_results