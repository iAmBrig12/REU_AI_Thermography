import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

class ThermDataset(Dataset):

    def __init__(self, file_path, direction=0, noise_scale=0, spec_scale=10**12):

        # read excel file and load row data into variables
        df = pd.read_excel(file_path)
        temp = df.iloc[:,:11].values
        spec = df.iloc[:,11:].values

        # scale spectrum data
        spec = spec.multiply(spec_scale)

        if noise_scale > 0:
            if direction:
                temp = np.random.normal(loc=0, scale=noise_scale, size=temp.size)
            else:
                spec = np.random.normal(loc=0, scale=noise_scale, size=spec.size)

        if direction:
            self.X_train = torch.tensor(temp, dtype=torch.float32)
            self.y_train = torch.tensor(spec)
        else:
            self.X_train = torch.tensor(spec, dtype=torch.float32)
            self.y_train = torch.tensor(temp, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]