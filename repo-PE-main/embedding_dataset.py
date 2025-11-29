import os
import torch
from torch.utils.data import Dataset

class HLSpectrogramDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.index = [] # List of (file_path, batch_idx)

        # Pre-index all batches across all files each file should have (B, J, C, H, W) so we want to select batches individually
        for file_name in sorted(os.listdir(data_dir)):
            if not file_name.endswith('.pt'):
                continue
            file_path = os.path.join(data_dir, file_name)
            try: 
                data = torch.load(file_path)['data']
                B = data.shape[0] # Number of batches
                for b in range(B):
                    self.index.append((file_path, b))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        file_path, batch_idx = self.index[idx]
        data = torch.load(file_path)['data']
        sample = data[batch_idx]  # shape: (J, C, F, T)

        return sample.float()  # shape: (10, 2, 80, 205)
