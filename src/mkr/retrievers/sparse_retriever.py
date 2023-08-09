import os
import json
import pickle
import numpy as np

from typing import List, Optional
from dataclasses import dataclass
from pythainlp.tokenize import word_tokenize
from mkr.utilities.general_utils import read_corpus
from rank_bm25 import BM25Okapi, BM25Plus, BM25L, BM25


@dataclass
class BM25Config:
    model_name: str
    tokenizer_name: str
    corpus_dir: str


class BM25SparseRetriever:
    def __init__(self, config: BM25Config, index: Optional[BM25] = None):        
        self.model_name = config.model_name
        self.tokenizer_name = config.tokenizer_name
        self.corpus_dir = config.corpus_dir

        self.corpus = read_corpus(self.corpus_dir)

        self.index = index
        if self.index is None:
            self.index = self._create_index(self.corpus)

    def _create_index(self, corpus: List[str]):
        # Tokenize corpus
        tokenized_corpus = [word_tokenize(doc, engine=self.tokenizer_name) for doc in corpus]
        # Create index
        if self.model_name == "bm25_okapi":
            index = BM25Okapi(tokenized_corpus)
        elif self.model_name == "bm25_plus":
            index = BM25Plus(tokenized_corpus)
        elif self.model_name == "bm25_l":
            index = BM25L(tokenized_corpus)
        return index
    
    def save_index(self, index_dir: str):
        # Create index_dir if not exists
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        # Save index
        pickle.dump(self.index, open(os.path.join(index_dir, "index.pkl"), "wb"))
        # Save config
        config = {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "corpus_dir": self.corpus_dir,
        }
        json.dump(config, open(os.path.join(index_dir, "config.json"), "w"))

    def __call__(self, query: str, top_k: int = 3):
        # Tokenize query
        tokenized_query = word_tokenize(query, engine=self.tokenizer_name)
        # Retrieve documents
        scores = self.index.get_scores(tokenized_query)
        sorted_scores = np.argsort(scores)[::-1]
        # Get top-k results
        returns = []
        for idx in sorted_scores[:top_k]:
            returns.append({"doc_id": idx, "doc_text": self.corpus[idx], "score": scores[idx]})
        return returns

    @classmethod
    def from_preindexed(cls, index_dir: str):
        # Check if index_dir exists
        assert os.path.exists(index_dir), f"Index directory not found: {index_dir}"
        # Check if relevant files exist
        assert os.path.exists(os.path.join(index_dir, "index.pkl")), f"Index file not found: {os.path.join(index_dir, 'index.pkl')}"
        assert os.path.exists(os.path.join(index_dir, "config.json")), f"Config file not found: {os.path.join(index_dir, 'config.json')}"

        # Load index
        index = pickle.load(open(os.path.join(index_dir, "index.pkl"), "rb"))
        # Load config
        config = json.load(open(os.path.join(index_dir, "config.json"), "r"))
        return cls(
            config=config,
            index=index,
        )