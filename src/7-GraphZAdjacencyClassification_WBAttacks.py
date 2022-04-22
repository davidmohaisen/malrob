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
from art.attacks import CarliniLInfMethod
from art.classifiers import KerasClassifier

f = open("../Pickles/Graph/DataAdjacency","rb")
x_train,y_train,x_test,y_test = pickle.load(f)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)




print("==========================================================================")
print("LR")
print("==========================================================================")
x_test_new_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
f = open("../model/Graph/AdjacencyBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test_new_, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
x_test_new_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
f = open("../model/Graph/AdjacencyBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_new_, y_test)
print("The RF score is:",score)

print("==========================================================================")


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test_new = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

json_file = open('../model/Graph/AdjacencyBaselineCNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Graph/AdjacencyBaselineCNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test, verbose=1)
print('Test accuracy:', scores[1])




classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)


predictions = classifier.predict(x_test_new)
# print(predictions)
# print(y_test)
# exit()
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


attack = CarliniLInfMethod(classifier=classifier,max_iter=100, eps=1.0,learning_rate=1.0)
x_train_adv = attack.generate(x=x_test_new)
f = open("../Pickles/Graph/AdjacencyAdversarial","wb")
pickle.dump(x_train_adv,f)

print(x_train_adv[0])
# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_train_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print(x_train_adv.shape)




y_test = np.argmax(predictions, axis=1)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
print("==========================================================================")
print("LR")
print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
f = open("../model/Graph/AdjacencyBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Graph/AdjacencyBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test, y_test)
print("The RF score is:",score)

print("==========================================================================")
