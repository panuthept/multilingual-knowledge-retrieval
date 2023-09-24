from typing import List
from torch import Tensor
from torch.functional import F
from mkr.models.baseclass import SentenceEncoder
from transformers import AutoTokenizer, AutoModel
from mkr.resources.resource_manager import ResourceManager


class mE5SentenceEncoder(SentenceEncoder):
    def __init__(self, model_name: str = "mE5_base"):
        assert model_name in self.available_models, f"Unknown model name: {model_name}"

        self.resource_manager = ResourceManager()
        self.tokenizer = AutoTokenizer.from_pretrained(self.resource_manager.get_encoder_path(model_name))
        self.model = AutoModel.from_pretrained(self.resource_manager.get_encoder_path(model_name))

    def _average_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def _encode(self, queries: List[str]) -> Tensor:
        inputs = self.tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = self._average_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def _encode_queries(self, queries: List[str]) -> Tensor:
        queries = [f"query: {query}" for query in queries]
        return self._encode(queries)
    
    def _encode_passages(self, passages: List[str]) -> Tensor:
        passages = [f"passage: {passage}" for passage in passages]
        return self._encode(passages)
    
    @property
    def available_models(self):
        return ["mE5_base", "mE5_small", "mE5_large"]
    

if __name__ == "__main__":
    encoder = mE5SentenceEncoder("mE5_small")

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_queries(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode_passages(english_sentences)
    print(en_emb.shape)