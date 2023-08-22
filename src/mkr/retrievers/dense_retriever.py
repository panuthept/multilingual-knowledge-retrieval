import os
import math
import json
# import faiss
import pickle
import chromadb
# from tqdm import trange
from tqdm import tqdm
# from faiss import IndexFlat
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from mkr.encoders.mUSE import mUSESentenceEncoder
from mkr.retrievers.baseclass import Retriever, RetrieverOutput
from mkr.utilities.general_utils import read_corpus, normalize_score, readline_corpus


@dataclass
class DenseRetrieverConfig:
    model_name: str
    database_dir: str
    corpus_dirs: dict = field(default_factory=dict)  # {corpus_name: corpus_dir, ...}


class DenseRetriever(Retriever):
    def __init__(self, config: DenseRetrieverConfig, auto_indexing: bool = True):
        self.model_name = config.model_name
        self.database_dir = config.database_dir
        self.corpus_dirs = config.corpus_dirs

        self.encoder = self._load_encoder(self.model_name)
        self.client = chromadb.PersistentClient(path=self.database_dir)
        # self.client = chromadb.Client()

        self.vector_collections = {}
        if len(self.corpus_dirs) > 0 and auto_indexing:
            for corpus_name, corpus_dir in self.corpus_dirs.items():
                self._index_corpus(corpus_name, corpus_dir)

    def _load_encoder(self, model_name: str):
        # Load encoder
        if model_name == "mUSE":
            encoder = mUSESentenceEncoder()
        else:
            raise ValueError(f"Unknown encoder: {model_name}")
        return encoder

    def _index_corpus(self, corpus_name: str, corpus_dir: str):
        self.vector_collections[corpus_name] = self.client.get_or_create_collection(name=corpus_name, metadata={"hnsw:space": "cosine"}, embedding_function=self.encoder)
        for data in tqdm(readline_corpus(corpus_dir)):
            self.vector_collections[corpus_name].upsert(
                documents=[data["content"]],
                metadatas=[data["metadata"]],
                ids=[data["hash"]],
            )
    
    def add_corpus(self, corpus_name: str, corpus_dir: str):
        assert corpus_name not in self.client.list_collections(), f"Collection already exists: {corpus_name}"
        self._index_corpus(corpus_name, corpus_dir)
        self.corpus_dirs[corpus_name] = corpus_dir

    def update_corpus(self, corpus_name: str):
        assert corpus_name in self.client.list_collections(), f"Collection not found: {corpus_name}"
        self._index_corpus(corpus_name, self.corpus_dirs[corpus_name])

    def __call__(self, queries: List[str], top_k: int = 3, corpus_names: List[str] = None) -> RetrieverOutput:
        corpus_names = corpus_names if corpus_names is not None else list(self.client.list_collections())

        # Retrieve documents
        collection_resutls = {}
        for corpus_name in corpus_names:
            results = self.vector_collections[corpus_name].query(
                query_texts=queries,
                n_results=top_k,
            )
            # Normalize scores
            print(results)


    #     query_embeddings = self.encoder.encode_batch(queries, batch_size=batch_size)
    #     # Retrieve documents
    #     scoress, indicess = self.index.search(query_embeddings, k=top_k)
    #     # Get top-k results
    #     resultss = []
    #     for scores, indices in zip(scoress, indicess):
    #         results = {}
    #         for idx, score in zip(indices, scores):
    #             results[self.corpus[idx]["doc_id"]] = {
    #                 **self.corpus[idx],
    #                 "score": score,
    #             }
    #         resultss.append(results)
    #     # Normalize score
    #     resultss = normalize_score(resultss)
    #     return RetrieverOutput(
    #         queries=queries,
    #         resultss=resultss,
    #     )

    # @classmethod
    # def from_indexed(cls, index_dir: str):
    #     # Check if index_dir exists
    #     assert os.path.exists(index_dir), f"Index directory not found: {index_dir}"
    #     # Check if relevant files exist
    #     assert os.path.exists(os.path.join(index_dir, "index.pkl")), f"Index file not found: {os.path.join(index_dir, 'index.pkl')}"
    #     assert os.path.exists(os.path.join(index_dir, "config.json")), f"Config file not found: {os.path.join(index_dir, 'config.json')}"

    #     # Load index
    #     index = faiss.deserialize_index(pickle.load(open(os.path.join(index_dir, "index.pkl"), "rb")))
    #     # Load config
    #     config = DenseRetrieverConfig(**json.load(open(os.path.join(index_dir, "config.json"), "r")))
    #     return cls(
    #         config=config,
    #         index=index,
    #     )