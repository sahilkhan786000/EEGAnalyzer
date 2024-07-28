
from glob import glob
import itertools
import os
import mne
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam
#from keras.utils import to_categorical

data_path = "minor_project1"
#tasks = ['MATBmed','MATBeasy','RS_Beg_EC','RS_Beg_EO','RS_End_Ec','RS_End_EO']
tasks = ['Flanker','MATBdiff','MATBmed','MATBeasy','PVT','RS_Beg_EC','RS_Beg_EO','RS_End_Ec','RS_End_EO','oneBACK','twoBACK','zeroBACK']

n_subs = 20
n_sessions = 3

x = []
y = []
for sub_n, session_n in itertools.product(range(n_subs), range(n_sessions)):
  epochs_data = []
  labels = []
  for lab_idx, level in enumerate(tasks):
    sub = 'sub-{0:02d}'.format(sub_n+10)
    sess = f'ses-S{session_n+1}'
    path = os.path.join(os.path.join(data_path, sub), sess) + f'/eeg/{level}.set'
    data=mne.io.read_raw_eeglab(path, eog=(), preload=True, uint16_codec=None, verbose=None)
    epochs=mne.make_fixed_length_epochs(data, duration=1.0, preload=False, reject_by_annotation=True, proj=True, overlap=0.0, id=1, verbose=None)
    tmp = epochs.get_data()
    epochs_data.extend(tmp)
    labels.extend([lab_idx]*len(tmp))
  x.extend(epochs_data)
  y.extend(labels)

#numpy array
X = np.array(x)
Y = np.array(y)

from sklearn.utils import shuffle

X,Y=shuffle(X,Y,random_state=0)
X,Y=shuffle(X,Y,random_state=1)
X,Y=shuffle(X,Y,random_state=2)

print(X.shape)

print(X[0].shape)

print(X[0][0][0])

X=np.moveaxis(X,1,2)

print(X.shape)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

print(X_train)
# X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)

verbose, epochs, batch_size = 0, 10, 32
#model0 = Sequential()
#model0.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
#model0.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
#model0.add(Dropout(0.5))
#model0.add(MaxPooling1D(pool_size=2))
#model0.add(Flatten())
#model0.add(Dense(100, activation='relu'))
#model0.add(Dense(64, activation='softmax'))
#model0.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#model0.summary()

#history=model0.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test))
#print(model0.evaluate(X_test, y_test))

#import tensorflow as tf
#from tensorflow.keras.models import load_model
#model0.save('/home/201112250/model1.hdf5')
#model = load_model('/home/201112250/model1.hdf5')
#print(model.evaluate(X_test, y_test))


model1 = Sequential()
model1.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
#model1.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model1.add(Dropout(0.5))
model1.add(MaxPooling1D(pool_size=2))
model1.add(Flatten())
model1.add(Dense(100, activation='relu'))
model1.add(Dense(32, activation='softmax'))
model1.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model1.summary()

history=model1.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test))
print(model1.evaluate(X_test, y_test))

import tensorflow as tf
from tensorflow.keras.models import load_model
model1.save('/home/201112250/model2.hdf5')
modela = load_model('/home/201112250/model2.hdf5')
print(modela.evaluate(X_test, y_test))


model3 = Sequential()
model3.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]))
#model3.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model3.add(Dropout(0.2))
model3.add(MaxPooling1D(pool_size=3))
model3.add(Flatten())
model3.add(Dense(100, activation='relu'))
model3.add(Dense(64, activation='softmax'))
model3.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model3.summary()

history=model3.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test))
print(model3.evaluate(X_test, y_test))

import tensorflow as tf
from tensorflow.keras.models import load_model
model3.save('/home/201112250/model3.hdf5')
modelb = load_model('/home/201112250/model3.hdf5')
print(modelb.evaluate(X_test, y_test))


model4 = Sequential()
model4.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
model4.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model4.add(Dropout(0.2))
model4.add(MaxPooling1D(pool_size=2))
model4.add(Flatten())
model4.add(Dense(100, activation='relu'))
model4.add(Dense(64, activation='softmax'))
model4.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model4.summary()

history=model4.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(X_test, y_test))
print(model4.evaluate(X_test, y_test))

import tensorflow as tf
from tensorflow.keras.models import load_model
model4.save('/home/201112250/model4.hdf5')
modelc = load_model('/home/201112250/model4.hdf5')
print(modelc.evaluate(X_test, y_test))

