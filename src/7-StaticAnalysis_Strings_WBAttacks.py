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
from art.classifiers import KerasClassifier
from art.attacks import CarliniLInfMethod

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

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
print("==========================================================================")
print("LR")
print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
f = open("../model/Static/StringBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Static/StringBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test, y_test)
print("The RF score is:",score)

print("==========================================================================")
y_test_n = []
for i in range(len(y_test)):
    y_test_n.append(int(y_test[i]))
y_test = np.asarray(y_test_n)


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
# create model

json_file = open('../model/Static/StringBaselineDNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Static/StringBaselineDNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])



classifier = KerasClassifier(model=model, clip_values=(np.amin(x_test), np.amax(x_test)), use_logits=False)


predictions = classifier.predict(x_test)
# print(predictions)
# print(y_test)
# exit()
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


attack = CarliniLInfMethod(classifier=classifier,max_iter=100, eps=(0.03*np.amax(x_test)-np.amin(x_test)))
x_train_adv = attack.generate(x=x_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_train_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
print(x_train_adv.shape)




print(y_test[0],predictions[0])
for i in range(len(x_test[0])):
    print(x_test[0][i],x_train_adv[0][i])

y_test = np.argmax(predictions, axis=1)
y_test_n = []
for i in range(len(y_test)):
    y_test_n.append(str(y_test[i]))
y_test = np.asarray(y_test_n)

x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
print("==========================================================================")
print("LR")
print("==========================================================================")
# clf = LogisticRegression(random_state=0).fit(x_train, y_train)
f = open("../model/Static/StringBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Static/StringBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test, y_test)
print("The RF score is:",score)

print("==========================================================================")
