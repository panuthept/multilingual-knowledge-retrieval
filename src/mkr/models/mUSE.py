import math
import tensorflow_hub
import tensorflow_text  # Used by the mUSE model
import tensorflow as tf
from typing import List
from mkr.models.baseclass import SentenceEncoder
from mkr.resources.resource_manager import ResourceManager


class mUSESentenceEncoder(SentenceEncoder):
    def __init__(self, model_name: str = "mUSE"):
        self.resource_manager = ResourceManager()
        self.model = tensorflow_hub.load(self.resource_manager.get_encoder_path(model_name))

    def _encode(self, texts: List[str]) -> tf.Tensor:
        return self.model(texts)
    
    def _encode_queries(self, queries: List[str]) -> tf.Tensor:
        return self._encode(queries)
    
    def _encode_passages(self, passages: List[str]) -> tf.Tensor:
        return self._encode(passages)
    

if __name__ == "__main__":
    encoder = mUSESentenceEncoder()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_queries(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode_passages(english_sentences)
    print(en_emb.shape)