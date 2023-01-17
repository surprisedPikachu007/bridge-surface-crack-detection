from keras.models import load_model
import os
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

model = load_model('models.h5', compile=False)

# predict dataset
predict_dir = os.path.join('predict')
predict_dataset = tf.keras.preprocessing.image_dataset_from_directory(predict_dir)  

y_true = []
y_pred = []

for images, labels in predict_dataset:
    images = tf.image.resize(images, (160, 160))   
    predictions = model.predict_on_batch(images).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions.numpy())

print('Precision: ', precision_score(y_true, y_pred))
print('Recall: ', recall_score(y_true, y_pred))
print('F1: ', f1_score(y_true, y_pred))

# write the scores to score.txt
with open('judging_metrics_.txt', 'w') as f:
    f.write('predict dataset' + '\n')
    f.write('Precision: ' + str(precision_score(y_true, y_pred)) + '\n')
    f.write('Recall: ' + str(recall_score(y_true, y_pred)) + '\n')
    f.write('F1: ' + str(f1_score(y_true, y_pred)) + '\n\n')

# test dataset
test_dir = os.path.join('test')
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(test_dir)

y_true = []
y_pred = []

for images, labels in test_dataset:
    images = tf.image.resize(images, (160, 160))   
    predictions = model.predict_on_batch(images).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions.numpy())

print('Precision: ', precision_score(y_true, y_pred))
print('Recall: ', recall_score(y_true, y_pred))
print('F1: ', f1_score(y_true, y_pred))

# write the scores to score.txt
with open('judging_metrics_.txt', 'a') as f:
    f.write('test dataset' + '\n')
    f.write('Precision: ' + str(precision_score(y_true, y_pred)) + '\n')
    f.write('Recall: ' + str(recall_score(y_true, y_pred)) + '\n')
    f.write('F1: ' + str(f1_score(y_true, y_pred)))


