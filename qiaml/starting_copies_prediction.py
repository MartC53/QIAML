#from qiaml import decision_tree_trainer
import pandas as pd
import numpy as np
import pickle
import tifffile as tiff
import scipy.stats as stats
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

def prediction(testpath):
    all_para = ['image ID','Average Value','maximum','stdev','arr_median','skew_ar','kurto','kurto_deriv']
    xpara = ['stdev','skew_ar','kurto']
    ypara = 'image ID'
    with open('model.pkl', 'rb') as f:
        models = pickle.load(f)
    keys = [30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    #models,xpara,ypara,all_para,keys = decision_tree_trainer.get_models()
    for ncps in models:
        print(models[ncps])
        
    testinput = tif_to_mat(testpath)
    df = pd.DataFrame(testinput, columns = all_para)
    test_df = df
    R2ar = []
    for ncps in models:
        R2ar.append(modeltest(models[ncps],test_df,xpara,ypara)[1])
    predicted_cps = keys[np.argmax(R2ar)]
    return print('The starting N copies should be',predicted_cps)


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