import numpy as np
import tensorflow as tf
from Dataset_from_dir import Dataset_from_directory
import os

def CNN_Classifier_funct(test_dir, predict_dir, group):
	print(test_dir, predict_dir)
	train_ds, val_ds, pred_ds = Dataset_from_directory(900, 900, test_dir, predict_dir)
	class_names = train_ds.class_names
	print(class_names)

	normalization_layer = tf.keras.layers.Rescaling(1./255)

	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))


	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	num_classes = 6

	model = tf.keras.Sequential([
	  tf.keras.layers.Rescaling(1./255),
	  tf.keras.layers.Conv2D(32, 3, activation='relu'),
	  tf.keras.layers.MaxPooling2D(),
	  tf.keras.layers.Conv2D(32, 3, activation='relu'),
	  tf.keras.layers.MaxPooling2D(),
	  tf.keras.layers.Conv2D(32, 3, activation='relu'),
	  tf.keras.layers.MaxPooling2D(),
	  tf.keras.layers.Flatten(),
	  tf.keras.layers.Dense(128, activation='relu'),
	  tf.keras.layers.Dense(num_classes)
	])

	model.compile(
	  optimizer='adam',
	  loss=tf.keras.losses.MeanSquaredError())

	history = model.fit(
	  train_ds,
	  validation_data=val_ds,
	  epochs=35
	)
	
	with open("CNNoutput %s.txt" % group, "w") as text_file:
		text_file.write(model.summary())
	    text_file.write("final MSE for train is %.2f and for validation is %.2f" % 
	      (history.history['loss'][-1], history.history['val_loss'][-1]))
	os.mkdir('classifier %s' % group)
	model.save('classifier %s' % group)
