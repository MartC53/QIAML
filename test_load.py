import tensorflow as tf
from qiaml.get_data import Dataset_from_directory

group = 'A'
data_dir = './Datasets/3_split'
predict_dir = './Datasets/cropped_jpgs/3_split_test2'


# train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5)

# normalization_layer = tf.keras.layers.Rescaling(1./255) # normalizing pixel intensity, 8 bit camera
# pred_ds = pred_ds.map(lambda x, y: (normalization_layer(x), y)) # normalizing pixel intensity, 8 bit camera


model = tf.keras.models.load_model('classifier 3_split_final')
print(predict_ds)
# model.summary()