import tensorflow as tf
from qiaml.get_data import Dataset_from_directory
from PIL import Image

group = 'A'
data_dir = './Datasets/3_split'
predict_dir = './Datasets/cropped_jpgs/3_split_test2'


train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5)

normalization_layer = tf.keras.layers.Rescaling(1./255) # normalizing pixel intensity, 8 bit camera
pred_ds = pred_ds.map(lambda x, y: (normalization_layer(x), y)) # normalizing pixel intensity, 8 bit camera


model = tf.keras.models.load_model('classifier final_CNN4')
# model.summary()
# predict = model.predict(pred_ds)
# print(predict)

# test_accuracy = tf.keras.metrics.Accuracy()
# print(test_accuracy)
# test_loss = estimator.model.evaluate(train_ds)
# print("test set mse is %.2f" % test_loss)
# print('load complete')
# predictions = model.predict(pred_ds)

# matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
# predictions = model(pred_ds, training=False)

RMSprop = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001907, momentum=0.00082)

model.compile(
  optimizer=RMSprop,
  loss=tf.keras.losses.MeanSquaredError())

test_accuracy = tf.keras.metrics.Accuracy()
ds_test_batch = pred_ds

for (x, y) in ds_test_batch:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=True)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int64)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))