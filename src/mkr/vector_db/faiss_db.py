import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any


class FaissVectorCollection:
    def __init__(self, collection_path: str):
        self.collection_path = collection_path

        self.index = None
        self.ids = []
        self.contents = []
        self.metadatas = []
        self.content_count = 0

        self.unique_ids = set()

        if os.path.exists(self.collection_path):
            self.load()
        else:
            os.makedirs(self.collection_path)

    def _init_index(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(
            self, 
            ids: List[str], 
            contents: List[str], 
            vectors: np.ndarray, 
            metadatas: List[Dict[str, Any]],
        ):
        # Initial index if needed
        if self.index is None:
            embedding_dim = vectors[0].shape[-1]
            self._init_index(embedding_dim)

        # Add doc to index
        for content_id, content, vector, metadata in zip(ids, contents, vectors, metadatas):
            if content_id in self.unique_ids:
                continue
            self.ids.append(content_id)
            self.contents.append(content)
            self.metadatas.append(metadata)
            self.index.add(vector.reshape(1, -1))
            # Update content_count
            self.content_count += 1
            self.unique_ids.add(content_id)

    def search(self, query_vectors: np.ndarray, top_k: int = 3) -> List[List[Dict[str, Any]]]:
        lst_scores, lst_indices = self.index.search(query_vectors, k=top_k)
        lst_results = []
        for scores, indices in zip(lst_scores, lst_indices):
            results = []
            for score, index in zip(scores, indices):
                results.append({
                    "id": self.ids[index],
                    "content": self.contents[index],
                    "metadata": self.metadatas[index],
                    "score": score,
                })
            # Normalize scores
            max_score = max([result["score"] for result in results])
            for result in results:
                result["score"] = result["score"] / max_score
            lst_results.append(results)
        return lst_results
    
    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.collection_path):
            os.makedirs(self.collection_path)

        if self.content_count > 0:
            # Save index
            pickle.dump(faiss.serialize_index(self.index), open(os.path.join(self.collection_path, "index.pkl"), "wb"))
            # Save ids
            pickle.dump(self.ids, open(os.path.join(self.collection_path, "ids.pkl"), "wb"))
            # Save contents
            pickle.dump(self.contents, open(os.path.join(self.collection_path, "contents.pkl"), "wb"))
            # Save metadatas
            pickle.dump(self.metadatas, open(os.path.join(self.collection_path, "metadatas.pkl"), "wb"))
            # Save content_count
            pickle.dump(self.content_count, open(os.path.join(self.collection_path, "content_count.pkl"), "wb"))
        
    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.collection_path), f"Index directory not found: {self.collection_path}"

        if os.path.exists(os.path.join(self.collection_path, "index.pkl")):
            # Load index
            self.index = faiss.deserialize_index(pickle.load(open(os.path.join(self.collection_path, "index.pkl"), "rb")))
            # Load ids
            self.ids = pickle.load(open(os.path.join(self.collection_path, "ids.pkl"), "rb"))
            # Load contents
            self.contents = pickle.load(open(os.path.join(self.collection_path, "contents.pkl"), "rb"))
            # Load metadatas
            self.metadatas = pickle.load(open(os.path.join(self.collection_path, "metadatas.pkl"), "rb"))
            # Load content_count
            self.content_count = pickle.load(open(os.path.join(self.collection_path, "content_count.pkl"), "rb"))


class FaissVectorDB:
    def __init__(self, database_path: str):
        self.database_path = database_path

        self.collections = {}       # {name: FaissVectorCollection}
        self.collection_paths = {}  # {name: path}

        if os.path.exists(self.database_path):
            self.load()
        else:
            os.makedirs(self.database_path)

    def get_collection_names(self) -> List[str]:
        return list(self.collections.keys())

    def create_collection(self, name: str, force_create: bool = False):
        if name not in self.collections or force_create:
            self.collections[name] = FaissVectorCollection(os.path.join(self.database_path, name))
            self.collection_paths[name] = os.path.join(self.database_path, name)

    def get_collection(self, name: str, force_create: bool = False) -> FaissVectorCollection:
        if name not in self.collections or force_create:
            self.create_collection(name, force_create=force_create)
        return self.collections[name]

    def save(self):
        # Create save_dir if not exists
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        if len(self.collections) > 0:
            # Save collection paths
            pickle.dump(self.collection_paths, open(os.path.join(self.database_path, "collection_paths.pkl"), "wb"))
            # Save collections
            for collection in self.collections.values():
                collection.save()

    def load(self):
        # Check if index_dir exists
        assert os.path.exists(self.database_path), f"Index directory not found: {self.database_path}"

        if os.path.exists(os.path.join(self.database_path, "collection_paths.pkl")):
            # Load collection paths
            self.collection_paths = pickle.load(open(os.path.join(self.database_path, "collection_paths.pkl"), "rb"))
            # Load collections
            for name, path in self.collection_paths.items():
                self.collections[name] = FaissVectorCollection(path)