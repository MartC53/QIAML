import tensorflow as tf
import os
from qiaml.get_data import Dataset_from_directory
import sys

def final_cnn_loop(group, data_dir, predict_dir):

  train_ds, val_ds, predict_dir = Dataset_from_directory(data_dir, predict_dir, 0.5)

  class_names = train_ds.class_names
  print(class_names)


  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  # running on swap cache (RAM) for faster performance
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  # running on swap cache (RAM) for faster performance

  num_classes = 3

  model = tf.keras.Sequential(
      [
        tf.keras.layers.Rescaling(1./255, input_shape=(900, 900, 3)),
        tf.keras.layers.Conv2D(8, 3, activation='linear'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='linear'),
        tf.keras.layers.Dense(num_classes)
        ]
  )


  RMSprop = tf.keras.optimizers.RMSprop(
      learning_rate=0.0001907,
      momentum=0.00082
  )

  model.compile(
    optimizer=RMSprop,
    loss=tf.keras.losses.MeanSquaredError())

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size=0,
    epochs=100
  )

  sys.stdout = open("%s.txt" % group, "w")
  model.summary()
  print("final MSE for train is %.2f and for validation is %.2f" %
        (history.history['loss'][-1], history.history['val_loss'][-1]))
  sys.stdout.close()
  os.mkdir('%s' % group)
  model.save('%s' % group, save_traces=False)

# group = 'Cut'
# data_dir = './cut_to_4/3_split_test'
# predict_dir = './cut_to_4/3_split'

# final_cnn_loop(group, data_dir, predict_dir)

group = 'Cutflip'
data_dir ='./cut_to_4_flip/3_split'
predict_dir = './cut_to_4_flip/3_split_test'
final_cnn_loop(group, data_dir, predict_dir)
