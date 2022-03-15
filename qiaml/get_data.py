# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:52:41 2022

@author: Colem
"""
import numpy as np
import pandas as pd


"""
Data sets All and AB are two part needing to be concatenated, all other data
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
        data = pd.DataFrame(get_data_array(Dataset), columns=[
                            'Images', 'Copy number'])
        return data

