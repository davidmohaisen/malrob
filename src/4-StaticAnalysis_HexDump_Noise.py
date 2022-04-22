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

f = open("../Pickles/filteredFilesTrainTest","rb")
FilesBenignTrain, FilesMalwareTrain, FilesBenignTest, FilesMalwareTest = pickle.load(f)

pathBenign = "../linuxBenignStaticAnalysisResults/hexdump-New/"
pathMalware = "../malwareStaticAnalysisResults/hexdump/"

x_train = []
y_train = []
x_test = []
y_test = []
counter = 0
for filename in FilesBenignTrain:
    try:
        f = open(pathBenign+filename+".pickle","rb")
        x = pickle.load(f)
        x_train.append(x)
        y_train.append(0)
    except:
        counter += 1

for filename in FilesMalwareTrain:
    try:
        f = open(pathMalware+filename+".pickle","rb")
        x = pickle.load(f)
        x_train.append(x)
        y_train.append(1)
    except:
        counter += 1


for filename in FilesBenignTest:
    try:
        f = open(pathBenign+filename+".pickle","rb")
        x = pickle.load(f)
        x_test.append(x)
        y_test.append(0)
    except:
        counter += 1

for filename in FilesMalwareTest:
    try:
        f = open(pathMalware+filename+".pickle","rb")
        x = pickle.load(f)
        x_test.append(x)
        y_test.append(1)
    except:
        counter += 1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

results = []
for pert in np.arange(0.00,1.01,0.01):
    results.append([])
    noise = np.random.normal(pert, pert, x_test.shape)
    # x_test_new = x_test + x_test*noise
    x_test_new = x_test + (np.amax(x_train)-np.amin(x_train))*noise

    print("==========================================================================")
    print("LR")
    print("==========================================================================")
    # clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    f = open("../model/Static/HexDumpBaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test_new, y_test)
    print("The LR score is:",score)
    results[-1].append(score)

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    f = open("../model/Static/HexDumpBaselineRF","rb")
    clf = pickle.load(f)
    score = clf.score(x_test_new, y_test)
    print("The RF score is:",score)
    results[-1].append(score)

    print("==========================================================================")



    print("==========================================================================")
    print("Deep Learning")
    print("==========================================================================")
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test_new = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1],1))
    # create model

    json_file = open('../model/Static/HexDumpBaselineDNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Static/HexDumpBaselineDNN.h5")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    results[-1].append(scores[1])

f = open("/media/ahmed/HDD/ICDCS2021/Pickles/Noise/HexDump","wb")
pickle.dump(results,f)
for i in range(101):
    print(i, results[i])
