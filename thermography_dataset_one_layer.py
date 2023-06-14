import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ThermDataset(Dataset):

    def __init__(self, fp, noise_scale, layer, spec_scale=10**12):
        self.df = pd.read_excel(fp)
        self.wavelengths = self.df.columns[11:]
        
        if layer:
            temp = self.df.iloc[:,layer-1].values
        else:
            temp = self.df.iloc[:,:11].values

        spec = self.df.iloc[:,11:].values

        
        mean = spec.mean()
        noise = np.random.normal(0, mean*noise_scale, spec.shape)
        spec = spec + noise

        spec = spec * spec_scale
        
        if layer:
            self.temp = torch.tensor(temp, dtype=torch.float32).reshape(-1, 1)
        else:
            self.temp = torch.tensor(temp, dtype=torch.float32)
            
        self.spec = torch.tensor(spec, dtype=torch.float32)
        

        self.n_samples = self.df.shape[0]

    def __getitem__(self, index):
        return self.temp[index], self.spec[index]

    def __len__(self):
        return self.n_samples
    