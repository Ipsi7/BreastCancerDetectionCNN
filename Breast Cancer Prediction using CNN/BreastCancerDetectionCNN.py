import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn.datasets
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D

from tensorflow.keras.optimizers import Adam

print(tf.__version__)


#getting the datasets

breast_cancer = sklearn.datasets.load_breast_cancer()
print(breast_cancer)

X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)
print(X.shape, Y.shape)

#import data to Pandas data frame

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
print(data)
data['class'] = breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts()) #checking the class - how many benign and malignant
print(breast_cancer.target_names)  #will show the target names malignant and benign
print(data.groupby('class').mean()) #0 for malignant


#Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1) #stratify used for equal distribution
print(Y.shape,Y_train.shape, Y_test.shape)
print(Y.mean(),Y_train.mean(), Y_test.mean())
print(X.mean(),X_train.mean(), X_test.mean())
print(X.shape,X_train.shape, X_test.shape)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = X_train.reshape(455,30,1)
X_test = X_test.reshape(114,30,1)

epochs = 56
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape = (30,1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer=Adam(learning_rate=0.00056), loss = 'binary_crossentropy', metrics =['accuracy'])

history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=1)
