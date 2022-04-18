import tensorflow as tf
from qiaml.get_data import Dataset_from_directory



group = 'Cut'
data_dir = './cut_to_4/3_split_test'
predict_dir = './cut_to_4/3_split'

def predict(data_dir, predict_dir, group):
    train_ds, val_ds, pred_ds = Dataset_from_directory(data_dir, predict_dir, 0.5)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    # normalizing pixel intensity, 8 bit camera
    pred_ds = pred_ds.map(lambda x, y: (normalization_layer(x), y))
    # normalizing pixel intensity, 8 bit camera


    model = tf.keras.models.load_model(group)

    test_accuracy = tf.keras.metrics.Accuracy()
    ds_test_batch = pred_ds

    for (x, y) in ds_test_batch:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(x, training=True)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int64)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

predict(data_dir, predict_dir, group)

group = 'Cutflip'
data_dir ='./cut_to_4_flip/3_split'
predict_dir = './cut_to_4_flip/3_split_test'

predict(data_dir, predict_dir, group)

'''
Found 28 files belonging to 3 classes.
Using 14 files for validation.
Found 48 files belonging to 3 classes.
Test set accuracy: 33.333%
Found 192 files belonging to 3 classes.
Found 192 files belonging to 3 classes.
Using 96 files for validation.
Found 112 files belonging to 3 classes.
Test set accuracy: 28.571%
'''