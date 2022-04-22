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
from sklearn.decomposition import PCA
os.environ['KMP_DUPLICATE_LIB_OK']='True'

f = open("../stringTrain-test/PCA-Train-labels.pickle","rb")
x_train, y_train = pickle.load(f)
y_train = np.asarray(y_train)
y_train = np.where(y_train=="malware", 1, y_train)
y_train = np.where(y_train=="benign", 0, y_train)
f = open("../stringTrain-test/PCA-Test-labels.pickle","rb")
x_test, y_test = pickle.load(f)
y_test = np.asarray(y_test)
y_test = np.where(y_test=="malware", 1, y_test)
y_test = np.where(y_test=="benign", 0, y_test)
# x_train = x_train.todense()
# x_test = x_test.todense()



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
    f = open("../model/Static/StringBaselineLR","rb")
    clf = pickle.load(f)

    score = clf.score(x_test_new, y_test)
    print("The LR score is:",score)
    results[-1].append(score)

    print("==========================================================================")
    print("RF")
    print("==========================================================================")
    f = open("../model/Static/StringBaselineRF","rb")
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

    json_file = open('../model/Static/StringBaselineDNN.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("../model/Static/StringBaselineDNN.h5")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

    scores = model.evaluate(x_test_new, y_test, verbose=1)
    print('Test accuracy:', scores[1])
    results[-1].append(scores[1])

f = open("/media/ahmed/HDD/ICDCS2021/Pickles/Noise/String","wb")
pickle.dump(results,f)
for i in range(101):
    print(i, results[i])
