import torch 
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from typing import Tuple 

class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text=text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i, i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)


    def __getitem__(self, idx) -> Tuple[int, int]:
        return (self.input_ids[idx], self.target_ids[idx])