"""
Contains functions that interpret tiff and return models. 

Authors: Xuetao Ma
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import sklearn         
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

import scipy.stats as stats
from scipy.signal import savgol_filter

import pandas as pd    
import tifffile as tiff
import seaborn as sns
import matplotlib 
import pickle
import csv

def get_models():
    
    """
    combine all the functions above to get trained models and also display
    the R2 value of training.
    """
    pathes,imname_all,pathes_only,ncps = foldername_sorting('All')
    i = 0
    ii = 0
    dicts_train = {}
    dicts_test0 = {}
    dicts_test1 = {}
    test_pool = ['100,000 cps', '3,000 cps', '300 cps', '1,000 cps']
    for n in ncps:
        if n not in dicts_train: 
            dicts_train[n] = pathes[i]
            i += 1
        continue

        if n in test_pool:
            if n not in dicts_test0:
                dicts_test0[n] = pathes[i]
                i += 1
                continue

            if n not in dicts_test1:
                dicts_test1[n] = pathes[i]
                i += 1
    
    
    i = 0
    ii = 0
    dicts_train = {}
    dicts_test0 = {}
    dicts_test1 = {}
    train_pool = ['100,000 cps', '3,000 cps', '300 cps', '1,000 cps']
    test_pool = [1000,300,3000,100000]


    
    # take saved paths instead of the 
    traindicts={}
    with open('dicts_train_input.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            traindicts[int(row[0])]=row[1]
    
    dicts_train = traindicts
    
    for nc in ncps:
    # taking one path from the training pool. Only read if it's not established in the dictionary.
        n = change_name(nc)
        if n not in dicts_train:
            dicts_train[n] = pathes[i]
            i += 1
            continue

        # Same idea, but only taking when the number of copies matches the test pool defined above and was not taken by train pool. 
        if n in test_pool:

            if n not in dicts_test0:
                dicts_test0[n] = pathes[i]
                i += 1
                continue
            # only taking when the number of copies matches the test pool defined above and was not taken by other dicts.
            if n not in dicts_test1:
                dicts_test1[n] = pathes[i]
                i += 1
        else:
            i+=1
            pass

    print('The test datasets:',dicts_test0)
    print('The test datasets:',dicts_train)

    keys_test = sorted(dicts_test0.keys())
    keys_train = sorted(dicts_train.keys())

    # loading the tiffs to pandas dataframe for next step
    trainingdfsets = []
    testingdfsets0 = []
    testingdfsets1 = []
    all_para = ['image ID','Average Value','maximum','stdev','arr_median','skew_ar','kurto','kurto_deriv']

    #Loading for training 
    for key in keys_train:
        mat_train = tif_to_mat(dicts_train[key])
        df = pd.DataFrame(mat_train, columns = all_para)
        trainingdfsets.append(df)

    #loading for testing
    for key in keys_test:
        mat_test0 = tif_to_mat(dicts_test0[key])
        df = pd.DataFrame(mat_test0, columns = all_para)
        testingdfsets0.append(df)

        mat_test1 = tif_to_mat(dicts_test1[key])
        df = pd.DataFrame(mat_test1, columns = all_para)
        testingdfsets1.append(df)

        
    #training models
    models = {}
    #xpara = ['Average Value','maximum','stdev','arr_median','skew_ar','kurto','kurto_deriv']
    #xpara = ['Average Value','maximum','stdev','arr_median','skew_ar','kurto']
    xpara = ['stdev','skew_ar','kurto']

    ypara = 'image ID'
    i = 0
    for dtsets in trainingdfsets:
        model = makemodel(dtsets,xpara,ypara)
        models[keys_train[i]] = model
        i+=1

    with open('model.pkl','wb') as f:
        pickle.dump(models,f)
    return models,xpara,ypara,all_para,keys_train

def tif_to_mat(tiffpath):
    """
    Define a tif_to_mat function to get all statistical interpretations from the tiff file. 
    All statistical interpretations are 'image ID','Average Value','maximum','stdev','arr_median','skew_ar','kurto','kurto_deriv'
    
    Returns: a matrix of interpreted information.
    
    Parameters: Path to the tiff file.
    """
    im1 = tiff.imread(tiffpath)
    layer,xmax,ymax = im1.shape
    array_avg = []
    arr_max = []
    arr_min = []
    arr_median = []
    
    skew_ar = []
    kurto_ar = []
    
    stv_ar = []
    harm_mean_ar = []
    xmax = xmax - 1
    ymax = ymax - 1
    
    for ii in im1:
        ii = ii.astype('float64')
        
        tl_corner = (ii[0,0])
        br_corner = (ii[xmax,ymax])
        bl_corner = (ii[xmax,0])
        tr_corner = (ii[0,ymax])
        
        
        normal_param = 4/(tl_corner+br_corner+bl_corner+tr_corner)
        ii = ii / normal_param
        
        ii = ii - im1[0]
        
        arr_max.append(ii.max())
        array_avg.append(ii.mean())
        arr_median.append(np.median(ii))
        allstv = np.std(ii)
        stv_ar.append(allstv)
        #statistics.covariance
        #harm_mean_ar.append(harm_mean(ii))
        skew_ar.append(stats.skew(ii.reshape(-1,1))[0])
        kurto_ar.append(stats.kurtosis(ii.reshape(-1,1))[0])
    
    kurto_smo = savgol_filter(kurto_ar, 111, 5)
    kurto_diff = np.append(0,np.diff(kurto_smo))
    
    
    aindexs = np.arange(0,layer,1)
    marrays = [aindexs,array_avg,arr_max,stv_ar,arr_median,skew_ar,kurto_ar,kurto_diff]
    rsars = np.array(marrays)
    newshape = rsars.transpose()
    
    return newshape

def foldername_sorting(foldername):
    """
    Define a function to get all images under the 'All' folder, which is the folder that
    stores all tiff files return different level of paths for convenience.
    """
    list_names = os.listdir(foldername)
    datas = []
    pathes = []
    imname_all = []
    pathes_only = []
    names_inter = []
    i=0
    for names in list_names:
        imanames = os.listdir(foldername+'/'+names)
        for j in range(0,len(imanames)+1):
            try:
                pathes.append(foldername+'/'+names+'/'+imanames[j])
                imname_all.append(imanames[j])
                pathes_only.append(foldername+'/'+names)
                names_inter.append(names)
            except:
                pass
        #datas.append(cv2.imread(upper_foldername+foldername+'/'+names+'/'+imanames[0]))
        i+=1
    return pathes,imname_all,pathes_only,names_inter

def change_name(strname):
    """
    Change the keys from strain to numbers so we can sort the keys in dictionary.
    """
    xx1 = strname.replace(' cps','')
    xx2 = xx1.replace(',','')
    xx3 = int(xx2)
    return xx3

# make model
def makemodel(dfin,xpara,ypara):
    """
    Here is making the model based on the df inputed, xparameter and yparameter defined outside.
    
    Parameters: dataframe, x-parameters and y-parameter for training
    """
    X = dfin[xpara]
    y = dfin[ypara].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=98, shuffle=True)
    
    # A nested function for optimizing depths of the tree
    def regr_func (X_train,y_train,depths):
        regr = DecisionTreeRegressor(random_state=0,max_depth=depths)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        MSE = mean_squared_error(y_test, y_pred)
        R2 = r2_score(y_test, y_pred)
        return MSE,R2
    
    depths_array = np.arange(2,40,1)
    MSE_array = []
    R2_array = []

    for dps in depths_array:
        res = regr_func (X_train,y_train,dps)
        MSE_array.append(res[0])
        R2_array.append(res[1])
        
    MSEmin = np.argmin(MSE_array)
    R2max = np.argmax(R2_array)
    print('best depth is:',depths_array[MSEmin])
    
    # Training the model with the best depths found above
    regr = DecisionTreeRegressor(random_state=0,max_depth=depths_array[MSEmin])
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    print('R2 score is: ',R2)
    if R2 <0.9:
        print('Warning! Model not good')
    return regr

# test the model with the test datasets
def modeltest(regr,df,xpara,ypara):
    """
    Taking the regression model trained on train pool and use it for testing datasets. 
    Here is making the model based on the df inputed, xparameter and yparameter defined outside.
    """
    X = df[xpara]
    y = df[ypara].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=98, shuffle=True)
    y_pred = regr.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)
    return MSE,R2