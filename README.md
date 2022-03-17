# QIAML ![Test status](https://github.com/MartC53/QIAML/actions/workflows/python-package-conda.yml/badge.svg)
The goal of this project is to produce a model that accurately detects quantifies fluorescent nucleation sites of DNA amplification. Previous work in the Posner Research group has shown that these fluorescent nucleation sites correlate with the concentration of initial DNA concentration.

## Current Functionality:
- [x] Proof of concept
  - [x] Pre-process image by filter separation, auto-cropping, resizing, and pixel normalization 
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

Isothermal DNA amplification technologies, like recombinase polymerase amplification (RPA) have been put forth that are faster, cheaper, and more robust than qPCR. Yet isothermal amplification technologies are limited in their diagnostic capabilities as they are qualitative. However, **Recent studies in the Posner Lab Group have shown that RPA, an isothermal NAT, can also be quantitative through a spot nucleation to initial copy correlation** [1]. Similar nucleation site analysis has been applied to other assays and targets that used ML to produce a quantification model which rivals our linear range [2]. Thus, we are interested in applying ML models to improve the linear range of our assay.
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
Due to their size, all data set images are cropped and pre-processed using the ``auto_crop.py``. This function isolates the green fluorescent channel (the detection channel of our FAM fluorophore) applied an adaptive blurring and contrasting to the images to improve visual representation. The images are then cropped based on contours of the image, and saved as a 2D NumPy array in an .npy file. To access the images, the ``get_data.py`` file reads in the .npy and saves the data arrays as either a NumPy array or a pandas dataframe. The data available is triplicate images of the end point of the RPA reaction. To ensure the train and test splits contain all data in the range of interest (30-10,000 cp) the triplicate data is split into train and test groups. Here the data in AB,BC, or AC represent the train groups while A, B, or C are the test groups for the training sets BC, AC,AB respectively.
### Pre CNN image processing 
To improve model accuracy and run time the images are sized to 900x900 pixels. Additionally, the image pixel intensities are normalized on a range of [0,1] with a maximum intensity of 256 for the 8-bit camera these images were taken on.
### Model training  
The model used is a simple sequential model  containing five layers. The first layer is a rescaling of the data to normalize 8-bit pixel intensities on the range of [0,1]. The next layer is an 8 neuron deep 2D convolution layer. 2D convolution layers have been successfully used for image classification in the past [1]. To find the optimal number of neurons, the model was iteratively run with 128, 64, 32, 16, 8, and 4 neurons. Accuracy was defined by correct identification of test data. The model failed to run above 64 neurons and accuracy losses were observed at 32 neurons. The highest accuracy was observed with 8 and 4 neurons. The model is then flattened before being past to two hidden dense layers whose neuron depths were iteratively determined as above. 

The optimizer and activation function hyper parameters used in the layers were chosen by utilizing OPTUNA [2]. An OPTUNA optimization was run for 10 epochs and 10 trials. The optimizers scanned were RSME, Adam, and SDG. The activators scanned were exponential Linear Unit, rectified linear unit activation function, linear activation function, and Scaled Exponential Linear Unit. These functions were chosen as the data should be able to be forced into a linear fashion.

1. R. Chauhan, K. K. Ghanshala and R. C. Joshi, "Convolutional Neural Network (CNN) for Image Detection and Recognition," 2018 First International Conference on Secure Cyber Computing and Communication (ICSCCC), 2018, pp. 278-282, doi: 10.1109/ICSCCC.2018.8703316.
2. Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.

### Initial Input Correlation 
Describe results here

## Application Usage
There are a few requirements to run the current GUI. These are that the model needs to be downloaded and unzipped into your working directory. The model’s directory title should be “finalcnn”. This file can be downloaded [here](https://drive.google.com/drive/folders/1L-Yn5opjNfaOqfTedgdTPQEp0Bcc60Qi?usp=sharing), due to its size 1.1gb we are unable to upload it into our repository. The main branch should have the following directory structure:
```
.
├── Datasets
│   └── cropped_jpgs
│       └── A
├── Documentation
├── finalcnn
└── qiaml
    ├── Datasets
    └── tests_autocrop
```
Other requirements:
  - 16gb of ram is required to load the image dataframes. 
  - 16gb of ram is required to run the GUI.

To run the GUI, users should clone our repository, activate the provided environment, and run ```streamlit run streamlit_app.py``` from the terminal. The user should follow their command line instructions to open the GUI on their internet browser. 
The model does take a long time to load.
Once the model is loaded you can either select a file that is already been prepared or drop menu or import your own file. The widget will display the predicated range of the image.

## Current limitations
The desired model is a regressor, however, due to a lack of data this was not possible. Using simple regression models, our validation error would increase with each epoch which would eventually kill the model. We believe the validation error continued to increase due to the limited number of validation images available, a 20% validation split is only two images. Thus, there are no 1:1 validations available. What we believe to be happening is that images of one input copy number (ie 100 copies) was being validated against a different (1,000 copies) or no image whatsoever. 

To remedy this, we split the data into three classifications: High input copies of 3,000 and 10,000, Medium input copies of 300 and 1,000, and low input copies of 30 and 100 copies. By doing this we essentially double our number of training images- which are then split with a 50% validation split to allow there to be a 1:1 image validation.

Another issue with the current limited image library is that the testing and validation images are of the same initial input copy number as those of the test set, Meaning, the training, validation, and testing data all contain images of the same dilution factor. In the future, this model can be improved by utilizing validation and testing images of a different dilution factor as to thoroughly test the model on input copies dilutions not previously seen.

## Future plans
The model is not currently available to be installed via pip of conda due to a few limiting factors. The first limiting factor is that size of the datasets. The data sets and saved models are multiple gigabytes in size. These large sizes make storing this data on GitHub impractical. Future work in this space will require the data to be uploaded to a data sharing platform like Zenodo. Second, this work and work around this project area are under active development and should not be utilized for the diagnosis of disease and should not be used in the attempt to make a diagnosis. In the future, a setup.py file will be added once more training data is available, the datasets and model will be available for download and local training if desired or for modification of other assay types.

Future work plans to implement a one shot/few shot model to reduce training data dependency. We also plan to supplement our data by splitting the images from 900x900 to 225x225 to quadruple the data set size. Further data set enlargement may be available through image augmentation were random rotations and flips are applied to the data. Furthermore, images taken on a microscope (current images are phone based) can be used to supplement the data. This will require additional scrutiny for bias as the data sets from the phone and microscope are the same reaction system, they have vastly different camera sensors.
