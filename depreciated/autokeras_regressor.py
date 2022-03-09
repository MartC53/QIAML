import autokeras as ak

import tensorflow as tf


print(ak.__version__)


img_height = 900
img_width = 900

data_dir = ('BC')
predict_dir = ('A')

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


# Initialize the image regressor.
reg = ak.ImageRegressor()
# Feed the image regressor with training data.

reg.fit(train_ds, validation_data=val_ds, batch_size=2, epochs=20)

reg.export_model()
reg.save('autokeras_regressor')

# # Predict with the best model.
# predicted_y = reg.predict(x_test)
# print(predicted_y)


# # Evaluate the best model with testing data.
# print(reg.evaluate(x_test, y_test))
