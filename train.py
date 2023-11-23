import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import RobustScaler
from thermography_model import Net

# Parameters for Neural Network
args = {'lr':0.01,
        'train epochs':5000,
        }

data_fp = sys.argv[1]

df = pd.read_excel(data_fp)

# layer data
y = df.filter(regex='layer')

# spectrum data
X = df.iloc[:,len(y.columns):]

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# convert data to tensors
X_train_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y.values, dtype=torch.float32)
    
# define loss function
criterion = nn.L1Loss()

# instantiate model
model = Net(X_train_tensor.size()[1], y_train_tensor.size()[1], [30, 15])

# define optimizer
optimizer = torch.optim.Rprop(model.parameters(), lr=args['lr'])

# Neural Network Training Loop
best_loss = np.inf
best_epoch = 0
train_losses = []

print('epoch'.ljust(6) + '| loss' + '\n------|-------------')

for epoch in range(args['train epochs']):
    # forward pass
    outputs = model(X_train_tensor)

    # get loss
    loss = criterion(outputs, y_train_tensor)

    # update and backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # add losses to the list for tracking
    train_losses.append(loss.item())
    
    # check if it is the best loss
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_epoch = epoch+1

    if (epoch+1) % int(args['train epochs']/10) == 0:
        print(f'{epoch+1}'.ljust(6) + f'|   {loss:.4f}')

print(f'\nbest training loss: {best_loss:.3f} in epoch {best_epoch}\n')  

# save model
model_fp = 'trained_model.pth' if len(sys.argv) < 3 else sys.argv[2]
torch.save(model.state_dict(), model_fp)
