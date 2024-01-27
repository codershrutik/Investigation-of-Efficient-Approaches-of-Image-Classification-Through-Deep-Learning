import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
(X_train,y_train),(X_test,y_test) = datasets.mnist.load_data() X_test.shape
X_train.shape
y_train[:5]
y_train = y_train.reshape(-1,)
y_train[:5]
y_test = y_test.reshape(-1,)
classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] def plot_sample(X,y,index):
plt.figure(figsize=(15,2)) plt.imshow(X[index]) plt.xlabel(classes[y[index]])
plot_sample(X_train,y_train,98) X_train = X_train/255.0
X_test = X_test/255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
model.fit(x_train, y_train, epochs=10)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32') / 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10) test_loss, test_acc = model.evaluate(x_test, y_test) print('Test accuracy:', test_acc)
