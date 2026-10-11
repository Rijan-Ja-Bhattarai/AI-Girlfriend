import os 
import tiktoken
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
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)


    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.input_ids[idx], self.target_ids[idx])

def create_dataloader(text: str, max_length: int = 256, stride: int = 128, shuffle: bool = True, 
                    drop_last: bool = True, num_workers: int = ((os.cpu_count() or 1) // 2), batch_size: int = 4) -> DataLoader:
    dataset = GPTDataset(text=text, tokenizer=Tokenizer(), 
                         max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers, 
    )
    return dataloader