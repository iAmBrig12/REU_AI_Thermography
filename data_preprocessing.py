import pandas as pd
import torch
import numpy as np

def get_data(file_path='combined_data.xlsx', direction=0, spec_scale=10**12, noise_scale=0.01):
    df = pd.read_excel(file_path)
    temp = df.iloc[:,:11].values
    spec = df.iloc[:,11:].values

    spec = spec * spec_scale

    if noise_scale > 0:
        if direction:
            mean = temp.mean()
            noise = np.random.normal(0, mean*noise_scale, temp.shape)
            temp = temp + noise
        else:
            mean = spec.mean()
            noise = np.random.normal(0, mean*noise_scale, spec.shape)
            spec = spec + noise

    if direction:
        X = torch.tensor(temp, dtype=torch.float32)
        y = torch.tensor(spec, dtype=torch.float32)
    else:
        X = torch.tensor(spec, dtype=torch.float32)
        y = torch.tensor(temp, dtype=torch.float32)

    return X, y