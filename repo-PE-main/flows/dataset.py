import os
import torch
from torch.utils.data import Dataset



class SpectrogramParamDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith('.pt')])

        self.load_data()

    def __len__(self):
        return dataset.shape[0]


    def load_data(self):
        dataset = []
        theta = []
        for f in file_paths:
            data = torch.load(file_paths[0])
            dataset.append(data['data'])
            theta.append(data['theta'])
            
        self.dataset = torch.cat(dataset, dim=0)
        self.theta = torch.cat(theta, dim=0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # return theta and data
        return (
            self.theta[idx].to(device=device),
            self.dataset[idx].to(device=device))
        
        


    

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.file_paths[idx])
        data = sample['data']         # shape: [J, 2, F, T]
        params = sample['params']     # shape: [P] or [1, P]

        if isinstance(params, torch.Tensor) and params.ndim == 2:
            params = params[0]  # [P]

        return data.float(), params.float()]

    def load_model(path):
        encoder1 = Encoder(embedding_dim)
        encoder2 = Encoder(embedding_dim)

        # Load the checkpoint
        checkpoint = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        # Load weights
        encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
        encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

        encoder1.eval()
        encoder2.eval()