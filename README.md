#  Machine Learning for Depth Thermography: Predicting Volumetric Temperature Distributions from Thermal-Emission Spectral Data

## Abstract
Predicting temperature distributions beneath the surface of objects is of high interest for a variety of science and engineering applications. Here, we develop a machine learning model built from a technique called depth thermography, based on infrared thermal spectrum data that can remotely determine volumetric temperature distributions. Currently, depth thermography uses physical equations to make its predictions—however, these equations are highly noise-sensitive. To combat this issue, we have developed a shallow neural network that can make accurate decisions with noisy input data. The models in this repository are tuned for three different media: Fused Silica, Gallium Nitride (GaN), and Indium antimonide (InSb)

## Files and Usage
### Configuration Files: 
fused_silica.json, GaN_config.json, InSb_config.json

These files contain the neural network model training parameters as well as the necessary file paths for training and testing. Any adjustments made to the training data path and model parameters will be made here.



### tandem_final.ipynb
#### description: Combines spectrum to temperature network with a temperature to spectrum network


### individual_layer.ipynb
#### description: Feature-selected network that predicts each temperature layer individually


### Old Code Folder: 
### description: Original attempt at Thermography project by other people

### Underdeveloped Attempts Folder: 
### description: This project has many attempts at different network topologies and approaches. These attempts were not nearly as developed as the final models. 

#### data Folder: 
### description: This contains all of the data we had been given; however, the latest data we used was the data_3nm.xlsx file that is not in this folder.

 
