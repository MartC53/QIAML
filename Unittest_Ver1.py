import cv2
import numpy as np
import unittest
#from qiaml import *
#import qiaml
from qiaml.autocrop import crop
from qiaml.get_data import get_data_array



"""
This test file tests the TestCrop function and get_data_array function. Please run it under QIAML/qiaml folder.
"""

class TestCrop(unittest.TestCase):
    def test_smoke(self):
        crop('test_image.jpg')
        return
    
    def test_oneshot(self):
        with self.assertRaises(AssertionError):
            crop(1)
        return
    
    def test_oneshot1(self):
        with self.assertRaises(AssertionError):
            crop()
        return
    
    def test_oneshot2(self):
        with self.assertRaises(AssertionError):
            crop('test_image2.jpg')
        return
    
    
class Test_get_data_array(unittest.TestCase):
    def test_smoke(self):
        get_data_array('All')
        
    def test_oneshot(self):
        with self.assertRaises(AssertionError):
            get_data_array('D')
        return
        
    def test_oneshot2(self):
        with self.assertRaises(AssertionError):
            get_data_array()
        return
    

class Test_tiffsize(unittest.TestCase):

    #We can test if the input tiff images have same size
    def test_failure(self):
        tiffpath1 = 'All/30 cps/4-14-21 Expt 2R.tif'
        tiffpath2 = 'All/30 cps/4-14-21 Expt 2R.tif'
        newshape1 = tif_to_mat(tiffpath1)
        newshape2 = tif_to_mat(tiffpath2)
        assert np.shape(newshape1) == np.shape(newshape2), 'The images have different size'
        

class Test_lowestmse(unittest.TestCase):

    #We can test if the max_depth value used is the most accurate one
    def test_failure(self):
        depths_array = np.arange(2,40,2)
        MSE_array = []
        R2_array = []

        for dps in depths_array:
            res = regr_func (X_train,y_train,dps)
            MSE_array.append(res[0])
            R2_array.append(res[1])
        assert min(MSE_array) < max(MSE_array), 'The max_depth value was not correct'
        
        
class Test_pathsize(unittest.TestCase):

    #We can test if the input path size is correct
    def test_failure(self):
        def foldername_sorting(foldername):
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
                i+=1
            return pathes,imname_all,pathes_only,names_inter
        assert len(pathes) == len(pathes_only), 'The path numbers are incorrect'
        
        
class Test_modeltest(unittest.TestCase):

    #We can test if the modeltest & modelmake function is working properly
    def test_modeltest(self):
        def modeltest(regr,path2):
            mat2 = tif_to_mat(path2)
            df2 = pd.DataFrame(mat2, columns = ['image ID','Average Value','maximum','arr_min','meandiff','stdev'])
            X2 = df2.loc[:, df.columns!='image ID']
            y2 = df2['image ID'].values.reshape(-1, 1)
            X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.9, random_state=98, shuffle=True)
            y_pred2 = regr.predict(X_test2)
            MSE = mean_squared_error(y_test2, y_pred2)
            R2 = r2_score(y_test2, y_pred2)
            return MSE,R2

        def makemodel(path):
            mat = tif_to_mat(path)
            df = pd.DataFrame(mat, columns = ['image ID','Average Value','maximum','arr_min','meandiff','stdev'])
            X = df.loc[:, df.columns!='image ID']
    
            y = df['image ID'].values.reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=98, shuffle=True)
            def regr_func (X_train,y_train,depths):
                regr = DecisionTreeRegressor(random_state=0,max_depth=depths)
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)
                MSE = mean_squared_error(y_test, y_pred)
                R2 = r2_score(y_test, y_pred)
 
                return MSE,R2
            depths_array = np.arange(2,40,2)
            MSE_array = []
            R2_array = []
 
            for dps in depths_array:
                res = regr_func (X_train,y_train,dps)
                MSE_array.append(res[0])
                R2_array.append(res[1])
            MSEmin = np.argmin(MSE_array)
            R2max = np.argmax(R2_array)
    
            regr = DecisionTreeRegressor(random_state=0,max_depth=depths_array[MSEmin])
            regr.fit(X_train, y_train)
            y_pred = regr.predict(X_test)
            MSE = mean_squared_error(y_test, y_pred)
            R2 = r2_score(y_test, y_pred)
            print('R2 score is: ',R2)
            if R2 <0.9:
                print('Warning! Model not good')
            return regr
        
        
        assert len(modeltest(model,testpath[0])) =! 1, 'The path model making and test process are incorrect'
        
        
