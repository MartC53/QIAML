import tensorflow as tf
from get_data import get_data_array

predict = get_data_array('A')
model = tf.keras.models.load_model('classifier 3_split')
model.summary()

# model.predict()

# test_loss = estimator.model.evaluate(train_ds)
# print("test set mse is %.2f" % test_loss)
# print('load complete')
# predictions = model.predict(pred_ds)

# matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
