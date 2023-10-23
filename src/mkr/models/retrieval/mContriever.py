import torch
from typing import List
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from mkr.models.retrieval.baseclass import SentenceEncoder
from mkr.resources.resource_manager import ResourceManager


class mContrieverSentenceEncoder(SentenceEncoder):
    def __init__(self, model_name: str = "mContriever"):
        assert model_name in self.available_models, f"Unknown model name: {model_name}"

        self.resource_manager = ResourceManager()
        self.tokenizer = AutoTokenizer.from_pretrained(self.resource_manager.get_encoder_path(model_name))
        self.model = AutoModel.from_pretrained(self.resource_manager.get_encoder_path(model_name))
        self.model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def _mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def _encode(self, queries: List[str]) -> Tensor:
        inputs = self.tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors="pt")
        # Use GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = self._mean_pooling(outputs[0], inputs["attention_mask"])

        # Move to CPU if needed
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
        return embeddings
    
    def _encode_queries(self, queries: List[str]) -> Tensor:
        return self._encode(queries).detach()
    
    def _encode_passages(self, passages: List[str]) -> Tensor:
        return self._encode(passages).detach()
    
    @property
    def available_models(self):
        return ["mContriever", "mContriever_msmarco"]
    

if __name__ == "__main__":
    encoder = mContrieverSentenceEncoder("mContriever_msmarco")

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_queries(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode_passages(english_sentences)
    print(en_emb.shape)