import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class ThermDataset(Dataset):

    def __init__(self, file_path):

        # read excel file and load row data into variables
        df = pd.read_excel(file_path)
        x = df.iloc[:,:11].values
        y = df.iloc[:,11:].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y


        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]