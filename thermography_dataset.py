import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ThermDataset(Dataset):

    def __init__(self, fp, noise_scale, tandem=0, direction=0, spec_scale=10**12):
        self.df = pd.read_excel(fp)
        temp = self.df.iloc[:,:11].values
        spec = self.df.iloc[:,11:].values

        if noise_scale:
            if direction:
                mean = temp.mean()
                noise = np.random.normal(0, mean*noise_scale, temp.shape)
                self.temp = temp + noise
            else:
                mean = spec.mean()
                noise = np.random.normal(0, mean*noise_scale, spec.shape)
                spec = spec + noise

        if tandem:
            spec = spec * spec_scale
            self.x = torch.tensor(spec, dtype=torch.float32)
            self.y = torch.tensor(temp, dtype=torch.float32)
        else:
            scaler = StandardScaler()
            
            if direction:
                spec = spec * spec_scale
                temp = scaler.fit_transform(self.temp)
                self.x = torch.tensor(temp, dtype=torch.float32)
                self.y = torch.tensor(spec, dtype=torch.float32)
            else:
                spec = scaler.fit_transform(spec)
                self.x = torch.tensor(spec, dtype=torch.float32)
                self.y = torch.tensor(temp, dtype=torch.float32)

        self.n_samples = self.df.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    