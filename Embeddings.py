import torch 
from torch import nn 
from torch.nn.modules.sparse import Embedding
from Tokenizer import Tokenizer

def token_embedding(emedding_dim: int = 256) -> Embedding:
    return nn.Embedding(num_embeddings=Tokenizer().vocab_size(), embedding_dim=emedding_dim)


def __pos_embedding_layer(context_length: int = 4, output_dim: int = 256) -> Embedding:
    return nn.Embedding(num_embeddings=context_length, embedding_dim=output_dim)

def pos_embedding(context_length: int = 4) -> torch.Tensor:
    embedding_layer = __pos_embedding_layer(context_length=context_length)
    return embedding_layer(torch.arange(context_length))

def input_vector(token_embedding: torch.Tensor, pos_embedding: torch.Tensor) -> torch.Tensor:
    return token_embedding + pos_embedding
