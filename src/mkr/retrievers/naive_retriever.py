import json
import numpy as np


from mkr.encoders.mUSE import mUSESentenceEncoder


class NaiveRetriever:
    def __init__(self, index_file: str, docs_file: str, encoder_name: str):
        # Load index
        self.index = np.load(index_file)
        # Load documents
        self.docs = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.docs.append(data["doc_text"])
        # Load encoder
        if encoder_name == "mUSE":
            self.encoder = mUSESentenceEncoder()
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

    def __call__(self, query: str, top_k: int = 3):
        # Encode query
        query_emb = self.encoder.encode(query)
        # Calculate cosine similarity
        scores = np.dot(query_emb, self.index.T).flatten()
        # Sort scores
        sorted_idx = np.argsort(scores)[::-1]
        # Return top k results
        results = []
        for idx in sorted_idx[:top_k]:
            results.append({"doc_id": idx, "doc_text": self.docs[idx], "score": scores[idx]})
        return results