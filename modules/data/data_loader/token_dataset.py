import numpy as np
import torch
from torch.utils.data import Dataset
    
class TokenDataset(Dataset):
    def __init__(self, data_path, seq_len):
        self.data = np.load(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start: start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, target_ids