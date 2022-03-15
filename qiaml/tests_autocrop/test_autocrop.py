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
        
        
        
        
