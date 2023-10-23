import torch
from typing import List
from torch import Tensor
from torch.functional import F
from mkr.models.retrieval.baseclass import SentenceEncoder
from transformers import AutoTokenizer, AutoModel
from mkr.resources.resource_manager import ResourceManager


class mDPRSentenceEncoder(SentenceEncoder):
    def __init__(self, model_name: str = "mDPR"):
        assert model_name in self.available_models, f"Unknown query model name: {model_name}"

        query_model_name = self.get_query_model_name(model_name)
        passage_model_name = self.get_passage_model_name(model_name)

        self.resource_manager = ResourceManager()
        
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.resource_manager.get_encoder_path(query_model_name))
        self.passage_tokenizer = AutoTokenizer.from_pretrained(self.resource_manager.get_encoder_path(passage_model_name))

        self.query_model = AutoModel.from_pretrained(self.resource_manager.get_encoder_path(query_model_name))
        self.passage_model = AutoModel.from_pretrained(self.resource_manager.get_encoder_path(passage_model_name))
        self.query_model.eval()
        self.passage_model.eval()

        # Use GPU if available
        if torch.cuda.is_available():
            self.query_model = self.query_model.cuda()
            self.passage_model = self.passage_model.cuda()
    
    def _encode_queries(self, queries: List[str]) -> Tensor:
        inputs = self.query_tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors="pt")
        # Use GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        outputs = self.query_model(**inputs)
        embeddings = outputs.pooler_output

        # Move to CPU if needed
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
        return embeddings.detach()
    
    def _encode_passages(self, passages: List[str]) -> Tensor:
        inputs = self.passage_tokenizer(passages, max_length=512, padding=True, truncation=True, return_tensors="pt")
        # Use GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        outputs = self.passage_model(**inputs)
        embeddings = outputs.pooler_output

        # Move to CPU if needed
        if torch.cuda.is_available():
            embeddings = embeddings.cpu()
        return embeddings.detach()
    
    def get_query_model_name(self, model_name):
        mapping = {
            "mDPR": "mDPR_query",
            "mDPR_msmarco": "mDPR_query_msmarco",
            "mDPR_tied": "mDPR_tied",
            "mDPR_tied_msmarco": "mDPR_tied_msmarco",
        }
        return mapping[model_name]
    
    def get_passage_model_name(self, model_name):
        mapping = {
            "mDPR": "mDPR_passage",
            "mDPR_msmarco": "mDPR_passage_msmarco",
            "mDPR_tied": "mDPR_tied",
            "mDPR_tied_msmarco": "mDPR_tied_msmarco",
        }
        return mapping[model_name]
    
    @property
    def available_models(self):
        return ["mDPR", "mDPR_msmarco", "mDPR_tied", "mDPR_tied_msmarco"]
    

if __name__ == "__main__":
    encoder = mDPRSentenceEncoder("mDPR")

    english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
    en_emb = encoder.encode_queries(english_sentences)
    print(en_emb.shape)
    en_emb = encoder.encode_passages(english_sentences)
    print(en_emb.shape)