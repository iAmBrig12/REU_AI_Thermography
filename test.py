import torch
import torch.nn as nn
from thermography_model import Net
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import random
import numpy as np
import json
import datetime

def divider(text="", char="=", divider_length=80):
    if not (text==""):
        text = ' ' + text + ' '
    return text.center(divider_length, char)

# load json config
config_path = sys.argv[1]
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

model_fp = config['file_paths']['model']
hidden_sizes = config['model_params']['hidden layers']
results_fp = config['file_paths']['results']


folder_path = sys.argv[2]
file_list = os.listdir(folder_path)

test_names = []
test_data = []
for filename in file_list:
    file_path = os.path.join(folder_path, filename)

    test_data.append(pd.read_excel(file_path))
    test_names.append(filename)

all_test_losses = []
out_text = ''

for name_index, df in enumerate(test_data):
    filename = test_names[name_index]
    
    # layer data
    y = df.filter(regex='layer')

    # spectrum data
    X = df.iloc[:,len(y.columns):]

    # convert data to tensors
    X_test = torch.tensor(X.values, dtype=torch.float32)
    y_test = torch.tensor(y.values, dtype=torch.float32)

    # load model
    model = Net(len(X.columns), len(y.columns), hidden_sizes)
    model.load_state_dict(torch.load(model_fp))


    # test model on validation data
    test_criterion = nn.L1Loss()
    test_losses = []
    average_overall_loss = 0

    out_text += f'Test file: {test_names[name_index]}'
    with torch.no_grad():
        model.eval()

        # overall predictions and loss
        pred = model(X_test)
        loss = test_criterion(pred, y_test)

        average_overall_loss = loss.item()
        
        out_text += f'\nAverage Loss: {loss.item()}\n'

        out_text += 'Layer | Loss' + '\n------|--------\n'

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
            out_text += f'{i}'.ljust(6) + f'| {loss.item():.4f}\n'

    out_text += '\n'

    # create folder to save visualizations
    folder_name = f'{filename}'[:-5] + '_results'

    # Get the current working directory
    current_directory = os.getcwd()

    # Create the full path for the new folder
    new_folder_path = os.path.join(current_directory, results_fp)
    new_folder_path = os.path.join(new_folder_path, folder_name)

    # Check if the folder already exists
    if not os.path.exists(new_folder_path):
        # Create the new folder
        os.makedirs(new_folder_path)

    # plot loss by layer
    plt.figure(figsize=(9,6))
    title = f'Average Loss Per Layer for {filename}'
    plt.title(title)
    plt.barh(range(1,len(y.columns)+1), test_losses, color='mediumorchid')
    plt.yticks(range(1,len(y.columns)+1))
    plt.xlabel("Temperature Loss (K)")
    plt.ylabel("Layer")
    plt.xlim(0,max(test_losses)+1)

    for i, loss in enumerate(test_losses):
        plt.text(loss, i + 1, f'{loss:.3f}', ha='left', va='center')

    plt.savefig(f'{new_folder_path}/{title}.png')
    plt.close()

    # Prediction visualization
    def plot_comparison(pred, actual, title):
        plt.figure(figsize=(12, 6))
        plt.title(title, fontsize=20)
        plt.xlabel("Layer", fontsize=18)
        plt.ylabel("Temperature (K)", fontsize=18)
        plt.plot([i+1 for i in range(len(y.columns))], actual, color='darkgray', marker='s', label='actual')
        plt.plot([i+1 for i in range(len(y.columns))], pred, color='mediumorchid', marker='o', linestyle=' ', label='predicted')
        plt.xticks(range(1,len(y.columns)+1), fontsize=16)
        plt.yticks(fontsize=16)     
        plt.legend()
        plt.savefig(f'{new_folder_path}/{title}.png')
        plt.close()
        

    # pick 5 random indices from the dataset and plot the comparison between predicted and actual values
    pred_df = pd.DataFrame(pred.numpy())

    indices = []
    for i in range(0,5):
        n = random.randint(0, len(y) - 1)
        indices.append(n)


    for name_index, i in enumerate(indices):
        a = y.iloc[i,:]
        p = pred_df.iloc[i,:]

        title = f"Temperature Predictions of Random Sample {name_index+1} for {filename}"
        plot_comparison(p, a, title)


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

    title = f"Temperature Predictions of Most Average Sample for {filename}"

    plot_comparison(p, a, title)

    all_test_losses.append(test_losses)


N = len(test_data[0].filter(regex='layer').columns)
ind = np.arange(N)
width = 0.15

plt.figure(figsize=(12,6))

bars = []
for num, lst in enumerate(all_test_losses):
    bars.append(plt.bar(ind+width*num, lst, width, label=test_names[num]))

plt.xticks(ind+width, range(1, N+1))
plt.title('Loss per Layer for all Files')
plt.ylabel('Test MAE Loss')
plt.xlabel('Layer')
plt.legend()


plt.savefig(f'{results_fp}/all_files_loss.png')

print(out_text)

with open(f'{results_fp}/test_log.txt', 'a') as f:
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(ct.center(80, '=') + '\n')
    
    f.write(out_text +'\n')
    
    f.write(
        f"""Configuration:
    Learning Rate; {config['training_params']['learning_rate']}
    Epochs:        {config['training_params']['num_epochs']}
    Hidden Layers: {hidden_sizes}\n\n""")