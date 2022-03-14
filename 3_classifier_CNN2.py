import numpy as np
import tensorflow as tf
import os
from get_data import Dataset_from_directory
import sys

group = '3_split_final'
data_dir = './Datasets/3_split/'
predict_dir = './Datasets/cropped_jpgs/3_split_test'


train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5)

class_names = train_ds.class_names
print(class_names)

# normalization_layer = tf.keras.layers.Rescaling(1./255) # normalizing pixel intensity, 8 bit camera

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # normalizing pixel intensity, 8 bit camera


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) #running on swap cache (RAM) for faster performance
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE) #running on swap cache (RAM) for faster performance

num_classes = 3


# 9
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
  tf.keras.layers.Conv2D(8, 3, activation='linear'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(16, activation='linear'),
  tf.keras.layers.Dense(num_classes)
])

# # 6
# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
#   tf.keras.layers.Conv2D(16, 3, activation='linear'),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(32, activation='linear'),
#   tf.keras.layers.Dense(num_classes)
# ])


# 5
# model = tf.keras.Sequential([
#   tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(64, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(128, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(256, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])


# model = tf.keras.Sequential([
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Conv2D(32, 3, activation='relu'),
#   tf.keras.layers.MaxPooling2D(),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dense(num_classes)
# ])

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
model.save('classifier %s' % group, include_optimizer=False, )