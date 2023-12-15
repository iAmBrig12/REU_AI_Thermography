import torch
from thermography_model import Net
from sklearn.preprocessing import RobustScaler
import pandas as pd
import sys
import json

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

input_data_fp = sys.argv[2]
input_data = pd.read_excel(input_data_fp)

scaler = RobustScaler()
input_data_scaled = scaler.fit_transform(input_data)

input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)

# load model
model = Net(input_size, output_size, hidden_sizes)
model.load_state_dict(torch.load(model_fp))

pred = model(input_tensor)

pred_df = pd.DataFrame(pred.numpy())

pred_df.to_excel(f'{input_data_fp[:-5]}_predictions.xlsx', index=False)
