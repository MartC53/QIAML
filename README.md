# QIAML
The goal of this project is to produce a model that accurately detects quantifies fluorescent nucleation sites of DNA amplification. Previous work in the Posner Research group has shown that these fluorescent nucleation sites correlate with the concentration of initial DNA concentration.

## Current Functionality:
- [ ] Proof of concept
  - [ ] Pre-process image by filter separation, auto-cropping, resizing, and pixel normalization 
  - [x] Load preprocessed image into Tensorflow tensor
  - [x] Produce CNN for image classification:
    - [x] Train model for image classification
    - [x] Sequential model
    - [x] 2D convolution
    - [x] Compile model
    - [x] Save and load model 
  - [x] Make predictions based on input pictures
    - [x] Visualize predictions and image on GUI
- [ ] Regressor model
  -  [ ] Investigate One-shot/few-shot models 
  -  [ ] Use time resolved data 

## Motivation
### Abstract
Nucleic Acid Tests (NATs) detect nucleic acid from disease and infection. Detection is based on amplifying low levels of DNA/RNA allowing for detection of a single strand of DNA/RNA. The gold standard for quantitative nucleic acid testing is quantitative polymerase chain reaction (qPCR). However, qPCR is:
* slow
* expensive 
* fragile 

Isothermal DNA amplification technologies, like recombinase polymerase amplification (RPA) have been put forth that are faster, cheaper, and more robust than qPCR. Yet isothermal amplification technologies are limited in their diagnostic capabilities as they are qualitative. However, **Recent studies in the Posner Lab Group have shown that RPA, an isothermal NAT, can also be quantitative through a spot nucleation to initial copy correlation** [1]. Similar nucleation site analysis has been applied to other assays and targets and used ML to produce a quantification model which rivals our linear range [2]. Thus, we are interested in applying ML models to improve the linear range of our assay.
1.  Quantitative Isothermal Amplification on Paper Membranes using Amplification Nucleation Site Analysis
Benjamin P. Sullivan, Yu-Shan Chou, Andrew T. Bender, Coleman D. Martin, Zoe G. Kaputa, Hugh March, Minyung Song, Jonathan D. Posner
bioRxiv 2022.01.11.475898; doi: https://doi.org/10.1101/2022.01.11.475898 
2. Membrane-Based In-Gel Loop-Mediated Isothermal Amplification (mgLAMP) System for SARS-CoV-2 Quantification in Environmental Waters
Yanzhe Zhu, Xunyi Wu, Alan Gu, Leopold Dobelle, Clément A. Cid, Jing Li, and Michael R. Hoffmann
Environmental Science & Technology 2022 56 (2), 862-873
DOI: 10.1021/acs.est.1c04623

For more information please see [Further details in the wiki](https://github.com/MartC53/QIAML/wiki/Further-details)


## Methods
### Import Images
Due to their size, all data set images are cropped and pre-processed using the ``auto_crop.py``. This function isolates the green fluorescent channel (the detection channel of our FAM fluorophore) applied an adaptive blurring and contrasting to the images to improve visual representation. The images are then cropped based on contours of the image, and saved as a 2D NumPy array in an .npy file. To access the images, the ``get_data.py`` file reads in the .npy and saves the data arrays as either a NumPy array or a pandas dataframe. The data available is triplicate images of the end point of the RPA reaction. To ensure the train and test splits contain all data in the range of interest (3-10,000 cp) the triplicate data is split into train and test groups. Here the data in AB,BC, or AC represent the train groups while A, B, or C are the test groups for the training sets BC, AC,AB respectively.
### Pre CNN image processing 
To improve model accuracy and run time the images are are sized to 900x900 pixels. Additionally, the image pixel intensities are normalized on a range of [0,1] with a maximum intensity of 256 for the 8-bit camera these images were taken on.
### Model training  

The model used is a simple sequential model  containing five layers. The first layer is a rescaling of the data to normalize 8-bit pixel intensities on the range of [0,1]. The next layer is an 8 neuron deep 2D convolution layer. 2D convolution layers have been successfully used for image classification in the past [1]. To find the optimal number of neurons, the model was iteratively run with 128, 64, 32, 16, 8, and 4 neurons. Accuracy was defined by correct identification of test data. The model failed to run above 64 neurons and accuracy losses were observed at 32 neurons. The highest accuracy was observed with 8 and 4 4 neurons. The model is then flattened before being pasted to two hidden dense layers whose neuron depths were iteratively determined as above 

The hyper parameters of the optimizer used and the the activations used in the layers were chosen by utilizing OPTUNA [2]. An OPTUNA optimization was run for 10 epochs and 10 trials. The optimizers scanned were RSME, Adam, and SDG. The activators scanned were exponential Linear Unit, rectified linear unit activation function, linear activation function, and Scaled Exponential Linear Unit. These functions were chosen as the data should be able to be forced into a linear fashion.

1. R. Chauhan, K. K. Ghanshala and R. C. Joshi, "Convolutional Neural Network (CNN) for Image Detection and Recognition," 2018 First International Conference on Secure Cyber Computing and Communication (ICSCCC), 2018, pp. 278-282, doi: 10.1109/ICSCCC.2018.8703316.
2. Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

### Inital Input Correlation 
Describe results here

## Usage
Code runs with...

# QIAML

Quantitative Isothermal Amplification Machine Learning
goals:
  - cnn-classification 
  - skikit leran 
  - tensor flow
  - patters-unsuperviiers
  -   intensity prediction 
  - supervised-features-target goal 
  - need x and target value for ML
  - classification- 
 what’s the problem 
 who are the users and skill level
 is the user a developer of diagnostics what to expect
 what makes this a success-linearizing the range of response
 how much data do we have
