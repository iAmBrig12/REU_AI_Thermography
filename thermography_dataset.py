import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ThermDataset(Dataset):

    def __init__(self, fp, direction, noise_scale):
        self.df = pd.read_excel(fp)
        self.temp = self.df.iloc[:,:11].values
        self.spec = self.df.iloc[:,11:].values

        if noise_scale:
            if direction:
                mean = self.temp.mean()
                noise = np.random.normal(0, mean*noise_scale, self.temp.shape)
                self.temp = self.temp + noise
            else:
                mean = self.spec.mean()
                noise = np.random.normal(0, mean*noise_scale, self.spec.shape)
                self.spec = self.spec + noise

        scaler = StandardScaler()
        
        if direction:
            self.temp = scaler.fit_transform(self.temp)
            self.x = torch.tensor(self.temp, dtype=torch.float32)
            self.y = torch.tensor(self.spec, dtype=torch.float32)
        else:
            self.spec = scaler.fit_transform(self.spec)
            self.x = torch.tensor(self.spec, dtype=torch.float32)
            self.y = torch.tensor(self.temp, dtype=torch.float32)

        self.n_samples = self.df.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    