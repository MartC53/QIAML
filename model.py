import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory

"""This will load the model and return model converted into probability resutls
"""
def loadmodel():
    
    model = tf.keras.models.load_model('classifier 3_split_final')
    #model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return model
   
def predict(img, model):
    img_resize = img.copy()
    img_resized = img_resize.resize((900,900))
    img_resized = (np.expand_dims(img_resized,0))
    predictions = model.predict(img_resized)
    return predictions
