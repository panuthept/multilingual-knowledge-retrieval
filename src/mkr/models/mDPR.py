import math

from typing import List, Optional
from mkr.models.baseclass import SentenceEncoderBase
from mkr.resources.resource_manager import ResourceManager


class mDPRSentenceEncoder(SentenceEncoderBase):
    def __init__(self, model_name: str = "mDPR"):
        self.resource_manager = ResourceManager()
        pass

    def encode(self, text: str):
        pass

    def encode_batch(self, texts: List[str], batch_size: Optional[int] = 32):
        pass
    

if __name__ == "__main__":
    encoder = mDPRSentenceEncoder()

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_batch(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode(english_sentences[0])
    print(en_emb.shape)