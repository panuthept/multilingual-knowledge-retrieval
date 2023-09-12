import math
import tensorflow_hub
import tensorflow_text  # Used by the mUSE model
import tensorflow as tf

from typing import List, Optional
from mkr.models.baseclass import SentenceEncoderBase
from mkr.resources.resource_manager import ResourceManager


class mUSESentenceEncoder(SentenceEncoderBase):
    def __init__(self, model_name: str = "mUSE"):
        self.resource_manager = ResourceManager()
        self.model = tensorflow_hub.load(self.resource_manager.get_encoder_path(model_name))

    def encode(self, text: str):
        return self.model(text).numpy().reshape(1, -1)

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = 32):
        embeddings = []
        batch_num = math.ceil(len(texts) / batch_size)
        for batch_idx in range(batch_num):
            batch_texts = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            embeddings.append(self.model(batch_texts))
        embeddings = tf.concat(embeddings, axis=0).numpy()
        return embeddings
    

if __name__ == "__main__":
    encoder = mUSESentenceEncoder()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_batch(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode(english_sentences[0])
    print(en_emb.shape)