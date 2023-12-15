#  Machine Learning for Depth Thermography: Predicting Volumetric Temperature Distributions from Thermal-Emission Spectral Data

## Abstract
Predicting temperature distributions beneath the surface of objects is of high interest for a variety of science and engineering applications. Here, we develop a machine learning model built from a technique called depth thermography, based on infrared thermal spectrum data that can remotely determine volumetric temperature distributions. Currently, depth thermography uses physical equations to make its predictions—however, these equations are highly noise-sensitive. To combat this issue, we have developed a shallow neural network that can make accurate decisions with noisy input data. The models in this repository are tuned for three different media: Fused Silica, Gallium Nitride (GaN), and Indium antimonide (InSb)

## Files
### Configuration Files: 
#### fused_silica.json, GaN_config.json, InSb_config.json

- Json configuration files containing the neural network model training parameters as well as the necessary file paths for training and testing. 
- Any adjustments made to the training data path and model parameters will be made here.

### Neural Network:
#### thermography_model.py
- Python script to build neural network
- The structure of the network is dynamic based on the configuration file provided to accomodate different media

### Model Training:
#### train.py
- Python script to train a model using a config file and training data file
- Trained model will be saved to a path designated by config file

#### fused_silica_train.xlsx, GaN_train.xlsx, InSb_train.xlsx
- Training files for models

### Model Testing: 
#### test.py
- Python script to test a model specified in the config file 
- It will test all files in a given folder
- Results are saved in Fused Silica results, GaN results, or InSb results, as specified in config file
- Testing results:
    - All Temperature Prediction for a Sample - Visualization of the predictions made for a random sample across all test files
    - Average Loss Per Layer for <test_file> - Visulization of the MAE loss per material layer for each test file
    - Temperature Predictions of a Random Sample for <test_file> - Visualization of predictions for a random sample made using each test file
    - <material> test losses - Excel file with testing results for each file

#### Fused Silica results, GaN results, InSb results
- Folders that testing results are saved to
- These folders must be made before testing and are specified in config files

#### fused_silica_gauss, GaN_gauss, InSb_gauss
- These folders contain testing data for each material
- Testing data has a Gaussian distribution of noise applied to the exact values computed using depth thermography equations
- Noise distributions are: exact, SNR=55, SNR=50, SNR=45, and SNR=35, where SNR=35 represents the highest amount of noise and SNR=50 being the closest to what is expected from experimental data

### Data Visualization:
#### data_visualization.ipynb
- Jupyter Notebook that visualizes the noise added to the data in test files

## Usage
python train.py <model_config>

python test.py <model_config> <test_folder> 