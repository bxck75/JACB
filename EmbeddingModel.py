import numpy as np
import faiss
from transformers import AutoModel, AutoTokenizer
from numpy.linalg import norm
from FaissStorage import LLMChatBot,AdvancedVectorStore
import torch
class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, texts: list, max_length: int = 512):
        # Tokenize the texts
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        # Get the embeddings from the model
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the embeddings
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()



if __name__ == "__main__":
    # Initialize the embedding model
    model_name = 'jinaai/jina-embeddings-v2-small-en'
    embedding_model = EmbeddingModel(model_name)

    # Sample texts
    texts = ['How is the weather today?', 'What is the current weather like today?']
    
    # Get embeddings
    embeddings = embedding_model.encode(texts)