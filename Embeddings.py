import torch 
from torch import nn 


def __pos_embedding_layer(context_length: int = 4, output_dim: int = 256) -> torch.nn.modules.sparse.Embedding:
    return nn.Embedding(num_embeddings=context_length, embedding_dim=output_dim)

def pos_embedding(context_length: int = 4) -> torch.Tensor:
    embedding_layer = __pos_embedding_layer(context_length=context_length)
    return embedding_layer(torch.arange(context_length))
    