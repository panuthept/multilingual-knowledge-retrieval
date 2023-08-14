import os
import math
import json
import faiss
import pickle
from tqdm import trange
from faiss import IndexFlat
from dataclasses import dataclass
from typing import List, Dict, Optional
from mkr.encoders.mUSE import mUSESentenceEncoder
from mkr.retrievers.baseclass import Retriever, RetrieverOutput
from mkr.utilities.general_utils import read_corpus, normalize_score


@dataclass
class DenseRetrieverConfig:
    model_name: str
    corpus_dir: str
    batch_size: int = 32


class DenseRetriever(Retriever):
    def __init__(self, config: DenseRetrieverConfig, index: Optional[IndexFlat] = None):        
        self.model_name = config.model_name
        self.corpus_dir = config.corpus_dir
        self.batch_size = config.batch_size

        self.encoder = self._load_encoder(self.model_name)
        self.corpus = read_corpus(self.corpus_dir)

        self.index = index
        if self.index is None:
            self.index = self._create_index(self.corpus, batch_size=self.batch_size)

    def _load_encoder(self, model_name: str):
        # Load encoder
        if model_name == "mUSE":
            encoder = mUSESentenceEncoder()
        else:
            raise ValueError(f"Unknown encoder: {model_name}")
        return encoder

    def _create_index(self, corpus: List[Dict[str, str]], batch_size: int = 32):
        index = None
        for batch_idx in trange(math.ceil(len(corpus) / batch_size)):
            batch = corpus[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            doc_texts = [doc["doc_text"] for doc in batch]
            corpus_embeddings = self.encoder.encode_batch(doc_texts, batch_size=batch_size)
            if index is None:
                # Create FAISS index
                embeddings_dim = corpus_embeddings.shape[-1]
                index = faiss.IndexFlatIP(embeddings_dim)
            index.add(corpus_embeddings)
        return index
    
    def save_index(self, index_dir: str):
        # Create index_dir if not exists
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)
        # Save index
        pickle.dump(faiss.serialize_index(self.index), open(os.path.join(index_dir, "index.pkl"), "wb"))
        # Save config
        config = {
            "model_name": self.model_name,
            "corpus_dir": self.corpus_dir,
            "batch_size": self.batch_size,
        }
        json.dump(config, open(os.path.join(index_dir, "config.json"), "w"))

    def __call__(self, queries: List[str], batch_size: int = 32, top_k: int = 3) -> RetrieverOutput:
        query_embeddings = self.encoder.encode_batch(queries, batch_size=batch_size)
        # Retrieve documents
        scoress, indicess = self.index.search(query_embeddings, k=top_k)
        # Get top-k results
        resultss = []
        for scores, indices in zip(scoress, indicess):
            results = {}
            for idx, score in zip(indices, scores):
                results[self.corpus[idx]["doc_id"]] = {
                    **self.corpus[idx],
                    "score": score,
                }
            resultss.append(results)
        # Normalize score
        resultss = normalize_score(resultss)
        return RetrieverOutput(
            queries=queries,
            resultss=resultss,
        )

    @classmethod
    def from_indexed(cls, index_dir: str):
        # Check if index_dir exists
        assert os.path.exists(index_dir), f"Index directory not found: {index_dir}"
        # Check if relevant files exist
        assert os.path.exists(os.path.join(index_dir, "index.pkl")), f"Index file not found: {os.path.join(index_dir, 'index.pkl')}"
        assert os.path.exists(os.path.join(index_dir, "config.json")), f"Config file not found: {os.path.join(index_dir, 'config.json')}"

        # Load index
        index = faiss.deserialize_index(pickle.load(open(os.path.join(index_dir, "index.pkl"), "rb")))
        # Load config
        config = DenseRetrieverConfig(**json.load(open(os.path.join(index_dir, "config.json"), "r")))
        return cls(
            config=config,
            index=index,
        )