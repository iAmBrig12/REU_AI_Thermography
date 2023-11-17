import torch
import torch.nn as nn
from thermography_model import Net
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np

data_fp = sys.argv[2]

df = pd.read_excel(data_fp)

# layer data
y = df.filter(regex='layer')

# spectrum data
X = df.iloc[:,len(y.columns):]

# convert data to tensors
X_test = torch.tensor(X.values, dtype=torch.float32)
y_test = torch.tensor(y.values, dtype=torch.float32)

# load model
model = Net(len(X.columns), len(y.columns))

model_fp = sys.argv[1]
model.load_state_dict(torch.load(model_fp))



# test model on validation data
test_criterion = nn.L1Loss()
test_losses = []
average_overall_loss = 0

print('Test Losses')
with torch.no_grad():
    model.eval()

    # overall predictions and loss
    pred = model(X_test)
    loss = test_criterion(pred, y_test)

    average_overall_loss = loss.item()
    
    print(f'Average Loss: {loss.item()}\n')

    # per layer loss based on previous predictions
    pred_layers = []
    for i in range(pred.size(1)):
        column = pred[:, i]
        pred_layers.append(column)

    actual_layers = []
    for i in range(y_test.size(1)):
        column = y_test[:, i]
        actual_layers.append(column)

    for i in range(y_test.size(1)):
        loss = test_criterion(pred_layers[i], actual_layers[i])
        test_losses.append(loss.item())
        print(f'Layer {i+1}: {loss.item()}')


# create folder to save visualizations
folder_name = f'{data_fp}'[:-5] + '_results'

# Get the current working directory
current_directory = os.getcwd()

# Create the full path for the new folder
new_folder_path = os.path.join(current_directory, folder_name)

# Check if the folder already exists
if not os.path.exists(new_folder_path):
    # Create the new folder
    os.makedirs(new_folder_path)


# plot loss by layer
plt.figure(figsize=(9,6))
title = f'Average Loss Per Layer for {data_fp}'
plt.title(title)
plt.barh(range(1,len(y.columns)+1), test_losses, color='mediumorchid')
plt.yticks(range(1,len(y.columns)+1))
plt.xlabel("Temperature Loss (K)")
plt.ylabel("Layer")
plt.xlim(0,max(test_losses)+1)

for i, loss in enumerate(test_losses):
    plt.text(loss, i + 1, f'{loss:.3f}', ha='left', va='center')

plt.savefig(f'{folder_name}/layer_loss.png')

# Prediction visualization
def plot_comparison(pred, actual, sample):
    plt.figure(figsize=(12, 6))
    title = f"Temperature Predictions for Sample {sample}"
    plt.title(title, fontsize=20)
    plt.xlabel("Silica Layer", fontsize=18)
    plt.ylabel("Temperature (K)", fontsize=18)
    plt.plot([i+1 for i in range(len(y.columns))], actual, color='darkgray', marker='s', label='actual')
    plt.plot([i+1 for i in range(len(y.columns))], pred, color='mediumorchid', marker='o', linestyle=' ', label='predicted')
    plt.xticks(range(1,len(y.columns)+1), fontsize=16)
    plt.yticks(fontsize=16)     
    plt.legend()
    plt.savefig(f'{folder_name}/{title}')
    

# pick 5 random indices from the dataset and plot the comparison between predicted and actual values
pred_df = pd.DataFrame(pred.numpy())

indices = []
for i in range(0,5):
    n = random.randint(0, len(y))
    indices.append(n)

for i in indices:
    a = y.iloc[i,:]
    p = pred_df.iloc[i,:]

    plot_comparison(p, a, i)


# plot the most average sample
loss_dif = np.inf
average_sample = 0
closest_loss = 0

for i in range(len(X_test)):
    x_row = X_test[i]
    y_row = y_test[i]

    pred = model(x_row)
    loss = test_criterion(pred, y_row)

    dif = abs(loss.item() - average_overall_loss)
    if dif < loss_dif:
        loss_dif = dif
        average_sample = i
        closest_loss = loss.item()

a = y.iloc[average_sample,:]
p = pred_df.iloc[average_sample,:]

plot_comparison(p, a, average_sample)