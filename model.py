import tensorflow as tf
import numpy as np


def loadmodel():
    """

    Returns
    -------
    model : to be used for testing unknown images

    """
    model = tf.keras.models.load_model('finalcnn')
    #model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    return model


def predict(img, model):
    """

    Parameters
    ----------
    img : PIL Image
    model: the imported tf.cnn model
    Returns
    -------
    predictions : array 
        The array contains the confidence level for the given catigory

    """
    img_resize = img.copy()
    img_resized = img_resize.resize((900, 900))
    img_resized = (np.expand_dims(img_resized, 0))
    predictions = model.predict(img_resized)
    return predictions

