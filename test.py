import torch
import torch.nn as nn
from thermography_model import Net
from sklearn.preprocessing import RobustScaler
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import json
import datetime

# creates a divider string for output
def divider(text="", char="=", divider_length=80):
    if not (text==""):
        text = ' ' + text + ' '
    return text.center(divider_length, char)


# prediction visualization function
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
    plt.savefig(f'{results_fp}/{title}.png')
    plt.close()


# load json config
config_path = sys.argv[1]
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

# model params
input_size = config['model_params']['input_size']
output_size = config['model_params']['output_size']
hidden_sizes = config['model_params']['hidden_layers']

# file paths
model_fp = config['file_paths']['model']
results_fp = config['file_paths']['results']


# get files for testing
folder_path = sys.argv[2]
file_list = os.listdir(folder_path)

test_data = []
for filename in file_list:
    file_path = os.path.join(folder_path, filename)

    test_data.append((filename, pd.read_excel(file_path)))


# load model
model = Net(input_size, output_size, hidden_sizes)
model.load_state_dict(torch.load(model_fp))


# variables for test results
all_test_losses = []
sample_index = 10
sample_actual = test_data[0][1].filter(regex='layer').iloc[sample_index,:]
sample_predictions = []
out_text = ''

# testing loop
for entry in test_data:
    filename = entry[0]
    df = entry[1]
    
    # layer data
    y = df.filter(regex='layer')

    # spectrum data
    X = df.iloc[:,len(y.columns):]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # convert data to tensors
    X_test = torch.tensor(X_scaled, dtype=torch.float32)
    y_test = torch.tensor(y.values, dtype=torch.float32)


    # test model on validation data
    test_criterion = nn.L1Loss()
    test_losses = []
    average_overall_loss = 0

    out_text += f'Test file: {filename}'
    with torch.no_grad():
        model.eval()

        # get predictions
        pred = model(X_test)

        # export pred to excel
        pred_df = pd.DataFrame(pred.numpy())
        pred_df.to_excel(f'{results_fp}/pred{filename[4:-5]}.xlsx', index=False)
        
        # calculate loss
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
            out_text += f'{i+1}'.ljust(6) + f'| {loss.item():.4f}\n'

    out_text += '\n'

    # Loss per layer visualization
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

    plt.savefig(f'{results_fp}/{title}.png')
    plt.close()


    # sample visualization
    sample_pred = pred_df.iloc[sample_index,:]

    title = f"Temperature Predictions of a Random Sample for {filename}"

    plot_comparison(sample_pred, sample_actual, title)

    sample_predictions.append(sample_pred)
    all_test_losses.append(test_losses)


# compare losses across files
N = len(test_data[0][1].filter(regex='layer').columns)
ind = np.arange(N)
width = 0.15

plt.figure(figsize=(12,6))

bars = []
for num, lst in enumerate(all_test_losses):
    bars.append(plt.bar(ind+width*num, lst, width, label=test_data[num][0]))

plt.xticks(ind+width, range(1, N+1))
plt.title('Loss per Layer for all Files')
plt.ylabel('Test MAE Loss')
plt.xlabel('Layer')
plt.legend()

plt.savefig(f'{results_fp}/all_files_loss.png')


# compare predictions for sample
plt.figure(figsize=(12, 6))
plt.title("All Temperature Predictions for a Sample", fontsize=20)
plt.xlabel("Layer", fontsize=18)
plt.ylabel("Temperature (K)", fontsize=18)
plt.plot([i+1 for i in range(output_size)], sample_actual, label='actual')

for idx in range(len(test_data)-1):
    plt.plot([i+1 for i in range(output_size)], sample_predictions[idx], label=test_data[idx][0])

plt.xticks(range(1, output_size+1), fontsize=16)
plt.yticks(fontsize=16)     
plt.legend()
plt.savefig(f'{results_fp}/All Temperature Predictions for a Sample.png')
plt.close()


# print output
print(out_text)

# write output to text file 
with open(f'{results_fp}/test_log.txt', 'a') as f:
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(ct.center(80, '=') + '\n')
    
    f.write(out_text +'\n')
    
    f.write(
        f"""Configuration:
    Learning Rate: {config['training_params']['learning_rate']}
    Epochs:        {config['training_params']['num_epochs']}
    Hidden Layers: {hidden_sizes}\n\n""")