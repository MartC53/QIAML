# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:52:41 2022

@author: Colem
"""
import os

import numpy as np
import pandas as pd
from tensorflow.keras.utils import image_dataset_from_directory

"""
Data sets All, AB, and AC are two part needing to be concatenated, all other data
sets are one piece 
"""



def get_data_array(Dataset):
    """
    
    Parameters
    ----------
    Dataset : str. Which data set desired: A,B,C,AB,AC,BC

    Returns
    -------
    data : array 
        Intexting the first image, array[i][j]-> i is the number of arrays in
        range of number of images, j=0 is eelecting image array, j=1 is the
        input copy number .
        
    """
    
    if Dataset == 'All':
        All_1 = np.load('./Datasets/QIAML_All_1_Data.npy', allow_pickle=True)
        All_2 = np.load('./Datasets/QIAML_All_2_Data.npy', allow_pickle=True)
        data = np.concatenate((All_1, All_2), axis=0)
        return data
    elif Dataset == 'AB':
        All_1 = np.load('./Datasets/QIAML_Data_AB_1.npy', allow_pickle=True)
        All_2 = np.load('./Datasets/QIAML_Data_AB_2.npy', allow_pickle=True)
        data = np.concatenate((All_1, All_2), axis=0)
        return data
    elif Dataset == 'A':
        data = np.load('./Datasets/QIAML_Data_A.npy', allow_pickle=True)
        return data
    elif Dataset == 'B':
        data = np.load('./Datasets/QIAML_Data_B.npy', allow_pickle=True)
        return data
    elif Dataset == 'C':
        data = np.load('./Datasets/QIAML_Data_C.npy', allow_pickle=True)
        return data
    elif Dataset == 'AC':
        All_1 = np.load('./Datasets/QIAML_Data_AC_1.npy', allow_pickle=True)
        All_2 = np.load('./Datasets/QIAML_Data_AC_2.npy', allow_pickle=True)
        data = np.concatenate((All_1, All_2), axis=0)
        return data
    elif Dataset == 'BC':
        data = np.load('./Datasets/QIAML_Data_BC.npy', allow_pickle=True)
        return data
    else:
        print('Error: Datasets limited to A, B, C, AB, AC, BC')

def get_data_df(Dataset):
    """
    
    Parameters
    ----------
    Dataset : str. Which data set desired: A,B,C,AB,AC,BC

    Returns
    -------
    data : pandas dataframe. 
        DESCRIPTION: dataframe with columns: images, Copy number

    """
    if Dataset == 'All':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'AB':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'A':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'B':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'C':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'AC':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    elif Dataset == 'BC':
        data = pd.DataFrame(get_data_array(Dataset), columns = ['Images', 'Copy number'])
        return data
    else:
        print('Error: Datasets limited to A, B, C, AB, AC, BC')
        
        
def get_data_tf_dataset(train, test):
	"""

	Parameters
	----------
	train : str. Which dataset desired: AB,AC,BC

	test : str. corrisponding testdata: A,B,C 

	Returns
	-------
	train_ds : tensorflow dataset for training. 
	pred_ds : tensorflow dataset for testing/predicting.
	Error: you need to download the data

	"""
	if os.path.exists('./Datasets/cropped_jpgs') == True:
		data_dir = './Datasets/cropped_jpgs/{train}'
		predit_dir = './Datasets/cropped_jpgs/{test}'
		Dataset_from_directory(data_dir, predict_dir)
	else:
		print('Error: Download the files from the shared drive link and store them in ./Datasets/cropped_jpgs')

def Dataset_from_directory(data_dir, predict_dir, split):
	"""
	    
	Parameters
	----------
	train : str. Which dataset desired: AB,AC,BC

	test : str. corrisponding testdata: A,B,C 
	
	split : float. Fraction of validation split (0.2 =20%)

	Returns
	-------
	train_ds : tensorflow dataset for training. 
	pred_ds : tensorflow dataset for testing/predicting.

	"""

	train_ds = image_dataset_from_directory(
	data_dir,
	seed=53,
	image_size=(900, 900))

	val_ds = image_dataset_from_directory(
		data_dir,
	validation_split=split,
	subset="validation",
	seed=53,
	image_size=(900, 900))

	pred_ds= image_dataset_from_directory(
		predict_dir,
		seed = 53,
		image_size=(900, 900))

	return train_ds, val_ds, pred_ds
