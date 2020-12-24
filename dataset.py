from torch.utils.data import Dataset

class ViolenceDataset(Dataset):
    def __init__(self, data, maxlen=128):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        log_S, label = self.data[idx]
        return log_S, label