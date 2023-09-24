import os
import json
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any
from mkr.utilities.general_utils import normalize_score


class AutoVectorSeachEngine:
    @classmethod
    def create_engine(cls, embeddings: np.ndarray, engine_name: str = "faiss"):
        if engine_name == "faiss":
            if embeddings.shape[0] < 1000000:
                # Using flat index on small dataset
                engine = faiss.IndexFlatIP(embeddings.shape[-1])
                engine.add(embeddings)
            else:
                # Using IVFFlat index on large dataset
                engine = faiss.IndexIVFFlat(
                    faiss.IndexFlatIP(embeddings.shape[-1]),
                    embeddings.shape[-1],
                    100,
                    faiss.METRIC_INNER_PRODUCT,
                )
                engine.train(embeddings)
                engine.add(embeddings)
        return engine


class VectorCollection:
    def __init__(self, collection_path: str, engine_name: str = "faiss",):
        self.collection_path = collection_path
        self.engine_name = engine_name
        # Initial parameters
        self.ids = []
        self.contents = []
        self.metadatas = []
        self.embeddings = None
        self.unique_ids = set()
        self.default_engine = None
        # Load parameters if exists
        if os.path.exists(self.collection_path):
            self.load()

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            vectors: np.ndarray, 
            metadatas: List[Dict[str, Any]],
        ):
        # Initial embeddings
        if self.embeddings is None:
            self.embeddings = np.zeros((1000, vectors.shape[-1]))

        # Add doc to index
        prev_idx = len(self.unique_ids)
        new_indices = []
        for index, (content_id, content, metadata) in enumerate(zip(ids, contents, metadatas)):
            if content_id in self.unique_ids:
                continue
            new_indices.append(index)
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            self.unique_ids.add(content_id)
        end_index = prev_idx + len(new_indices)

        # Add embeddings
        # If embeddings is full, resize it
        if end_index > self.embeddings.shape[0]:
            new_size = self.embeddings.shape[0] * 2
            self.embeddings.resize(new_size, self.embeddings.shape[-1])
        # Add new embeddings
        self.embeddings[prev_idx:end_index] = vectors[new_indices]

    def search(
            self, 
            query_vector: np.ndarray, 
            top_k: int = 3, 
            candidate_ids: List[str] = None,
        ) -> List[Dict[str, Any]]:
        # Get search engine
        if candidate_ids is not None:
            # If candidate_ids is provided, search only in candidate_ids
            # To do so, we need to create a new search engine
            candidate_indices = [self.ids.index(candidate_id) for candidate_id in candidate_ids]
            engine = AutoVectorSeachEngine.create_engine(self.embeddings[candidate_indices], self.engine_name)
        else:
            candidate_indices = np.arange(len(self.unique_ids))
            if self.default_engine is None:
                self.default_engine = AutoVectorSeachEngine.create_engine(self.embeddings, self.engine_name)
            engine = self.default_engine

        # Search
        lst_scores, lst_indices = engine.search(query_vector, k=top_k)
        results = []
        for score, cand_index in zip(lst_scores[0], lst_indices[0]):
            # Filter indices that exceed the number of documents
            if cand_index >= len(candidate_indices):
                continue
            real_index = candidate_indices[cand_index]
            results.append({
                "id": self.ids[real_index],
                "content": self.contents[real_index],
                "metadata": self.metadatas[real_index],
                "score": score,
            })
            
        # Normalize scores
        results = normalize_score(results)
        return results
    
    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)

        if len(self.unique_ids) > 0:
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "w", encoding="utf-8") as f:
                for content_id, content, metadata in zip(self.ids, self.contents, self.metadatas):
                    f.write(json.dumps({
                        "id": content_id,
                        "content": content,
                        "metadata": metadata,
                    }, ensure_ascii=False))
                    f.write("\n")
            # Save embeddings
            pickle.dump(self.embeddings, open(os.path.join(self.collection_path, "embeddings.pkl"), "wb"))
        
    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.collection_path), f"Index directory not found: {self.collection_path}"

        if os.path.exists(os.path.join(self.collection_path, "embeddings.pkl")):
            self.ids = []
            self.contents = []
            self.metadatas = []
            self.unique_ids = set()
            with open(os.path.join(self.collection_path, "corpus.jsonl"), "rb") as f:
                for line in f:
                    data = json.loads(line)
                    self.ids.append(data["id"])
                    self.contents.append(data["content"])
                    self.metadatas.append(data["metadata"])
                    self.unique_ids.add(data["id"])
            # Load embeddings
            self.embeddings = pickle.load(open(os.path.join(self.collection_path, "embeddings.pkl"), "rb"))
            # Load search engine
            self.default_engine = AutoVectorSeachEngine.create_engine(self.embeddings, self.engine_name)


class VectorDB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: VectorCollection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()

    def get_collection_names(self) -> List[str]:
        return list(self.collection_paths.keys())

    def create_or_get_collection(self, name: str) -> VectorCollection:
        if name not in self.collection_paths:
            self.collection_paths[name] = os.path.join(self.database_path, name)
            self.collections[name] = VectorCollection(self.collection_paths[name])
        if name not in self.collections:
            self.collections[name] = VectorCollection(self.collection_paths[name])
        return self.collections[name]
    
    def get_collection(self, name: str) -> VectorCollection:
        assert name in self.collection_paths, f"Collection not found: {name}"
        if name not in self.collections:
            self.collections[name] = VectorCollection(self.collection_paths[name])
        return self.collections[name]

    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        if len(self.collection_paths) > 0:
            # Save collection paths
            json.dump(self.collection_paths, open(os.path.join(self.database_path, "collection_paths.json"), "w"))
            # Save collections
            for collection in self.collections.values():
                collection.save()

    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.database_path), f"Index directory not found: {self.database_path}"

        if os.path.exists(os.path.join(self.database_path, "collection_paths.json")):
            # Load collection paths
            self.collection_paths = json.load(open(os.path.join(self.database_path, "collection_paths.json"), "r"))