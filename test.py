import torch
import torch.nn as nn
from thermography_model import Net
from sklearn.preprocessing import RobustScaler
import pandas as pd
import sys
import os
import json
import datetime

# creates a divider string for output
def divider(text="", char="=", divider_length=80):
    if not (text==""):
        text = ' ' + text + ' '
    return text.center(divider_length, char)

# check input
if len(sys.argv) < 4:
    print('Incorrect Format')
    print('python test.py <config_path> <model_name> <test_file_folder>')
    quit()

# load json config
config_path = sys.argv[1]
with open(config_path, 'r') as config_file:
    config = json.load(config_file)


# model params
input_size = config['model_params']['input_size']
output_size = config['model_params']['output_size']
hidden_sizes = config['model_params']['hidden_layers']


# file paths
model_name = sys.argv[2]
model_path = f'Models/{config["material"]}/{model_name}.pth'
test_folder_path = sys.argv[3]
file_list = os.listdir(test_folder_path)

test_data = []
for filename in file_list:
    file_path = os.path.join(test_folder_path, filename)
    test_data.append((filename, pd.read_excel(file_path)))


# load model
model = Net(input_size, output_size, hidden_sizes)
model.load_state_dict(torch.load(model_path))

# set up directory for test results
directory = model_name
parent_dir = f'Test Results/{config["material"]}'
save_path = os.path.join(parent_dir, directory)

try:
    os.makedirs(save_path, exist_ok=True)
    print(f'Directory {directory} created')
except OSError as error:
    print(f'Directory {directory} cannot be created')

# variables for testing
all_test_losses = []
cols = test_data[0][1].filter(regex='layer').columns
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
        
        # save predictions to folder
        pred_df = pd.DataFrame(pred.numpy(), columns=cols)
        pred_df.to_excel(f'Test Results/{config["material"]}/{model_name}/pred{filename[4:]}', index=False)
        
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

    all_test_losses.append(test_losses)


# export losses to excel
loss_df = pd.DataFrame(all_test_losses, columns=cols)
loss_df['test file'] = file_list
loss_df.to_excel(f'Test Results/{config["material"]}/{model_name}/test losses.xlsx', index=False)


# print output
print(out_text)

# write output to text file 
with open(f'Test Results/{config["material"]}/test_log.txt', 'a') as f:
    ct = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    f.write(ct.center(80, '=') + '\n')
    f.write(f'Model: {model_name}\n\n')
    
    f.write(out_text +'\n')
    
    f.write(
        f"""Configuration:
    Learning Rate: {config['training_params']['learning_rate']}
    Epochs:        {config['training_params']['num_epochs']}
    Hidden Layers: {hidden_sizes}\n\n""")