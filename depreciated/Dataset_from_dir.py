import numpy as np
import os
import tensorflow as tf


def Dataset_from_directory(img_height, img_width,
  data_dir, predict_dir):

  train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=53,
    image_size=(img_height, img_width))

  val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=53,
    image_size=(img_height, img_width))

  pred_ds= tf.keras.utils.image_dataset_from_directory(
  	predict_dir,
  	seed = 53,
  	image_size=(img_height, img_width))

  return train_ds, val_ds, pred_ds