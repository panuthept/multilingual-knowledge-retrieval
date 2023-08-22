import math
import tensorflow_hub
import tensorflow_text  # Used by the mUSE model
import tensorflow as tf

from typing import List, Optional
from mkr.encoders.baseclass import SentenceEncoderBase
from mkr.resources.resource_manager import ResourceManager


class mUSESentenceEncoder(SentenceEncoderBase):
    def __init__(self):
        self.resource_manager = ResourceManager()
        self.model = tensorflow_hub.load(self.resource_manager.get_encoder_path("mUSE"))

    def encode(self, text: str):
        return self.model(text).numpy()

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = 32):
        results = []
        batch_num = math.ceil(len(texts) / batch_size)
        for batch_idx in range(batch_num):
            batch_texts = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            results.append(self.model(batch_texts))
        results = tf.concat(results, axis=0)
        return results.numpy()
    
    def __call__(self, texts: List[str]):
        return self.encode_batch(texts)
    

if __name__ == "__main__":
    encoder = mUSESentenceEncoder()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_batch(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode(english_sentences[0])
    print(en_emb.shape)