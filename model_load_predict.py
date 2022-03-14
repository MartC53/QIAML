import tensorflow as tf
from qiaml.get_data import Dataset_from_directory


group = 'A'
data_dir = './Datasets/3_split'
predict_dir = './Datasets/cropped_jpgs/3_split_test2'


train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5)

normalization_layer = tf.keras.layers.Rescaling(1./255) # normalizing pixel intensity, 8 bit camera
pred_ds = pred_ds.map(lambda x, y: (normalization_layer(x), y)) # normalizing pixel intensity, 8 bit camera


model = tf.keras.models.load_model('classifier 3.14 Classifier')

test_accuracy = tf.keras.metrics.Accuracy()
ds_test_batch = pred_ds

for (x, y) in ds_test_batch:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=True)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int64)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))