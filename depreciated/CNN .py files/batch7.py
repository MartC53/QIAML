import numpy as np
import tensorflow as tf
import os
from qiaml.get_data import Dataset_from_directory
import sys

group = 'batch7' #Name for output directory and file
data_dir = './Datasets/3_split/' #input training data
predict_dir = './Datasets/cropped_jpgs/3_split_test' #input testing data, not used but required to use Dataset from directory 


train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5) #loading data

class_names = train_ds.class_names #double checking that class names are correct
print(class_names) #double checking that class names are correct


AUTOTUNE = 0
print(AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) #running on swap cache (RAM) for faster performance
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) #running on swap cache (RAM) for faster performance

num_classes = 3


model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
  tf.keras.layers.Conv2D(8, 3, activation='linear'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, activation='linear'),
  tf.keras.layers.Dense(num_classes)
])


RMSprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001907, momentum=0.00082)

model.compile(
  optimizer=RMSprop,
  loss=tf.keras.losses.MeanSquaredError())

history = model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=0,
  epochs=100
)

sys.stdout = open("3_CNNoutput %s.txt" % group, "w")
model.summary()
print("final MSE for train is %.2f and for validation is %.2f" % 
      (history.history['loss'][-1], history.history['val_loss'][-1]))
sys.stdout.close()
os.mkdir('classifier %s' % group)
model.save('classifier %s' % group)