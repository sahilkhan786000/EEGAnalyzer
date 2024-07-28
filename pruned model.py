
#!pip install mne

from glob import glob
import itertools
import tempfile
import os
import mne
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.optimizers import Adam
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
#from keras.utils import to_categorical
import tensorboard

data_path = "minor_project1"
#tasks = ['MATBmed','MATBeasy','RS_Beg_EC','RS_Beg_EO','RS_End_Ec','RS_End_EO']
tasks = ['Flanker','MATBdiff','MATBmed','MATBeasy','PVT','RS_Beg_EC','RS_Beg_EO','RS_End_Ec','RS_End_EO','oneBACK','twoBACK','zeroBACK']

n_subs = 1
n_sessions = 1

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

print(X)

print(X[0].shape)

print(X[0][0][0])

X=np.moveaxis(X,1,2)

print(X.shape)

print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

print(X_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=0)


prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude



verbose, epochs, batch_size = 0, 10, 32

steps_per_epoch = len(X_train) // batch_size
pruning_schedule= tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                               final_sparsity=0.8,
                                                               begin_step=0,
                                                               end_step=4000)


model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model_for_pruning = prune_low_magnitude(model, pruning_schedule=pruning_schedule)
model_for_pruning.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
model_for_pruning.summary()

#history=model_for_pruning.fit(X_train,y_train,epochs=epochs, batch_size=batch_size,validation_data=(X_val, y_val))

callbacks = [sparsity.UpdatePruningStep(),sparsity.PruningSummaries(log_dir="./logs", profile_batch=0)]
model_for_pruning.fit(X_train,y_train,batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks=callbacks)

print(model_for_pruning.evaluate(X_test, y_test,verbose))

#pd.DataFrame(history.history).plot(figsize=(8, 5))

#pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()

from tensorflow.keras.models import load_model
model_for_pruning.save('/home/201112250/prun2.h5')

import joblib
joblib.dump(model_for_pruning, "/home/201112250/prun2.pkl")
print("1");


