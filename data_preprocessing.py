import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_data(file_path='combined_data.xlsx', direction=0, spec_scale=10**12, noise_scale=0.01):
    df = pd.read_excel(file_path)
    temp = df.iloc[:,:11].values
    spec = df.iloc[:,11:].values

    #spec = spec * spec_scale

    if noise_scale > 0:
        if direction:
            mean = temp.mean()
            noise = np.random.normal(0, mean*noise_scale, temp.shape)
            temp = temp + noise
        else:
            mean = spec.mean()
            noise = np.random.normal(0, mean*noise_scale, spec.shape)
            spec = spec + noise

    scaler = StandardScaler()
    
    if direction:
        X = scaler.fit_transform(temp)
        y = spec
    else:
        X = scaler.fit_transform(spec)
        y = temp

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return X, y