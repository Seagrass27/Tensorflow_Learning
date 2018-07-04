from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

digits = load_digits() #1797个手写数字
fig = plt.figure(figsize=(3, 3))
plt.imshow(digits['images'][66], cmap="gray", interpolation='none')

from sklearn import svm
classifier = svm.SVC(gamma=0.001)
classifier.fit(digits.data, digits.target) # train;target是0-9的整数
predicted = classifier.predict(digits.data) # predict

import numpy as np
print(np.mean(digits.target == predicted))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)
print(np.mean(predicted == y_test))

# =============================================================================
# Do it in TensorFlow Learn, which is a higher API
# =============================================================================
from tensorflow.contrib import learn
import tensorflow as tf

n_classes = len(set(y_train))
#classifier needs to be told what types of features to expect
#to see feature column:https://www.tensorflow.org/get_started/feature_columns
classifier = learn.LinearClassifier(
        feature_columns=[tf.contrib.layers.real_valued_column(
                "", dimension=X_train.shape[1])], n_classes=n_classes)
classifier.fit(X_train, y_train, steps=10)

y_pred = classifier.predict(X_test) #返回一个generator

y_pred = np.array(list(y_pred))

from sklearn import metrics
print(metrics.classification_report(y_true=y_test, y_pred=y_pred))
