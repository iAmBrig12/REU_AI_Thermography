#  Machine Learning for Depth Thermography: Predicting Volumetric Temperature Distributions from Thermal-Emission Spectral Data

## Abstract
Predicting temperature distributions beneath the surface of objects is of high interest for a variety of science and engineering applications. Here, we develop a machine learning model built from a technique called depth thermography, based on infrared thermal spectrum data that can remotely determine volumetric temperature distributions. Currently, depth thermography uses physical equations to make its predictions—however, these equations are highly noise-sensitive. To combat this issue, we have developed a shallow neural network that can make accurate decisions with noisy input data. The models in this repository are tuned for three different media: Fused Silica, Gallium Nitride (GaN), and Indium antimonide (InSb)

## Files
### Configuration Files: 
#### fused_silica.json, GaN_config.json, InSb_config.json

- Json configuration files containing the neural network model training.
- Any adjustments made to the training data path and model parameters will be made here.

### Neural Network:
#### thermography_model.py
- Python script to build neural network.
- The structure of the network is dynamic based on the configuration file provided to accomodate different media.

### Model Training:
#### train.py
- Python script to train a model using a config file and training data file.
- Trained model state dictionary will be saved to a path designated by the user in the command line input.
- All models are saved in the Models folder under the material that it was trained for

### Model Testing: 
#### test.py
- Python script to test a model specified in the command line
- It will test all files in a folder given in the command line
- Results are saved in Test Results under the material specified in config file. Under the material folders are the results for individual models.
- Testing results:
    - Test Losses - MAE Loss calculations per test file for the given model
    - Predictions - predictions made using each test file for the given model to be used in further testing or visualization.

#### fused_silica_gauss, GaN_gauss, InSb_gauss
- These folders contain testing data for each material
- Testing data has a Gaussian distribution of noise applied to the exact values computed using depth thermography equations

### Data Visualization:
#### data_visualization.ipynb
- Jupyter Notebook that visualizes the noise added to the data in test files

## Usage
python train.py <config_path> <training_data_path> <model_name>

python test.py <config_path> <model_name> <test_file_folder>

### Ex. 
python train.py GaN_config.json GaN_training_data/exact.xlsx GaN_model_exact

python test.py GaN_config.json GaN_model_exact GaN_gauss
