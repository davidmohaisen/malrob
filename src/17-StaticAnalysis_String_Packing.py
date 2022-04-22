from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from keras.models import model_from_json
from PIL import Image
from sklearn.utils import shuffle
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import collections
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.decomposition import PCA
f = open("../Pickles/filteredFilesTrainTest","rb")
_, _, FilesBenignTest, FilesMalwareTest = pickle.load(f)


f = open("../stringTrain-test/PCA-Test-labels.pickle","rb")
x_test, y_test = pickle.load(f)

y_test = np.asarray(y_test)
y_test = np.where(y_test=="malware", 1, y_test)
y_test = np.where(y_test=="benign", 0, y_test)

# x_train = x_train.todense()
# x_test = x_test.todense()
# y_test_new = []
# for i in range(len(y_test)):
#     y_test_new.append(int(y_test[i]))
# y_test = np.asarray(y_test_new)

print(x_test.shape)
print(y_test.shape)


x_test_benign = []
y_test_benign = []
x_test_malware = []
y_test_malware = []
for i in range(len(x_test)):
    if y_test[i] == '0':
        x_test_benign.append(x_test[i])
        y_test_benign.append(y_test[i])
    else:
        x_test_malware.append(x_test[i])
        y_test_malware.append(y_test[i])

x_test_benign = np.asarray(x_test_benign)
y_test_benign = np.asarray(y_test_benign)
x_test_malware = np.asarray(x_test_malware)
y_test_malware = np.asarray(y_test_malware)


print("==========================================================================")
print("LR")
print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
f = open("../model/Static/StringBaselineLR","rb")
clf = pickle.load(f)


score = clf.score(x_test_benign, y_test_benign)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Static/StringBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_benign, y_test_benign)
print("The RF score is:",score)

print("==========================================================================")



print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test_new = np.reshape(x_test_benign,(x_test_benign.shape[0],x_test_benign.shape[1],1))
# create model
json_file = open('../model/Static/StringBaselineDNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Static/StringBaselineDNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test_benign, verbose=1)
print('Test accuracy:', scores[1])



print("==========================================================================")
print("LR")
print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
f = open("../model/Static/StringBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test_malware, y_test_malware)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Static/StringBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_malware, y_test_malware)
print("The RF score is:",score)

print("==========================================================================")



print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test_new = np.reshape(x_test_malware,(x_test_malware.shape[0],x_test_malware.shape[1],1))
# create model
json_file = open('../model/Static/StringBaselineDNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Static/StringBaselineDNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test_malware, verbose=1)
print('Test accuracy:', scores[1])



# exit()
