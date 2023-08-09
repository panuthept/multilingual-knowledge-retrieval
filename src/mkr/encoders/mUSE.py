import math
import tensorflow_hub
import tensorflow_text  # Used by the mUSE model
import tensorflow as tf

from tqdm import trange
from typing import List, Optional
from mkr.encoders.baseclass import SentenceEncoderBase


class mUSESentenceEncoder(SentenceEncoderBase):
    def __init__(self):
        # self.model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
        self.model = tensorflow_hub.load("./models/universal-sentence-encoder-multilingual_3")

    def encode(self, text: str):
        return self.model(text).numpy()

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = 32):
        results = []
        batch_num = math.ceil(len(texts) / batch_size)
        for batch_idx in trange(batch_num):
            batch_texts = texts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            results.append(self.model(batch_texts))
        results = tf.concat(results, axis=0)
        return results.numpy()
    

if __name__ == "__main__":
    encoder = mUSESentenceEncoder()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_batch(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode(english_sentences[0])
    print(en_emb.shape)