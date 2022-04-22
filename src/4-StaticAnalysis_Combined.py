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

x_train_HexDump = np.asarray(x_train)
y_train_HexDump = np.asarray(y_train)
x_test_HexDump = np.asarray(x_test)
y_test_HexDump = np.asarray(y_test)

print(x_train_HexDump.shape)
print(y_train_HexDump.shape)
print(x_test_HexDump.shape)
print(y_test_HexDump.shape)



f = open("../Pickles/Static/SectionsTrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../Pickles/Static/SectionsTrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_Sections = np.concatenate((x_train_1,x_train_2))
y_train_Sections = np.concatenate((y_train_1,y_train_2))
f = open("../Pickles/Static/SectionsTestData","rb")
x_test_Sections, y_test_Sections = pickle.load(f)

print(x_train_Sections.shape)
print(y_train_Sections.shape)
print(x_test_Sections.shape)
print(y_test_Sections.shape)



f = open("../Pickles/Static/SegmentsTrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../Pickles/Static/SegmentsTrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_Segments = np.concatenate((x_train_1,x_train_2))
y_train_Segments = np.concatenate((y_train_1,y_train_2))
f = open("../Pickles/Static/SegmentsTestData","rb")
x_test_Segments, y_test_Segments = pickle.load(f)

print(x_train_Segments.shape)
print(y_train_Segments.shape)
print(x_test_Segments.shape)
print(y_test_Segments.shape)






f = open("../stringTrain-test/PCA-Train-labels.pickle","rb")
x_train_String, y_train_String = pickle.load(f)
y_train_String = np.asarray(y_train_String)
y_train_String = np.where(y_train_String=="malware", 1, y_train_String)
y_train_String = np.where(y_train_String=="benign", 0, y_train_String)
f = open("../stringTrain-test/PCA-Test-labels.pickle","rb")
x_test_String, y_test_String = pickle.load(f)
y_test_String = np.asarray(y_test_String)
y_test_String = np.where(y_test_String=="malware", 1, y_test_String)
y_test_String = np.where(y_test_String=="benign", 0, y_test_String)
# x_train = x_train.todense()
# x_test = x_test.todense()



print(x_train_String.shape)
print(y_train_String.shape)
print(x_test_String.shape)
print(y_test_String.shape)







f = open("../Pickles/Static/SymbolsTrainDataP1","rb")
x_train_1, y_train_1 = pickle.load(f)
f = open("../Pickles/Static/SymbolsTrainDataP2","rb")
x_train_2, y_train_2 = pickle.load(f)
x_train_Symbols = np.concatenate((x_train_1,x_train_2))
y_train_Symbols = np.concatenate((y_train_1,y_train_2))
f = open("../Pickles/Static/SymbolsTestData","rb")
x_test_Symbols, y_test_Symbols = pickle.load(f)

print(x_train_Symbols.shape)
print(y_train_Symbols.shape)
print(x_test_Symbols.shape)
print(y_test_Symbols.shape)


x_train = np.concatenate((x_train_HexDump,x_train_Symbols,x_train_Sections,x_train_Segments), axis = 1)
x_test = np.concatenate((x_test_HexDump,x_test_Symbols,x_test_Sections,x_test_Segments), axis = 1)
y_train = y_train_Symbols
y_test = y_test_Symbols

print(x_train.shape)

print("==========================================================================")
print("LR")
print("==========================================================================")
clf = LogisticRegression(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The LR score is:",score)
f = open("../model/Static/CombinedBaselineLR","wb")
pickle.dump(clf,f)


print("==========================================================================")
print("RF")
print("==========================================================================")
clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
score = clf.score(x_test, y_test)
print("The RF score is:",score)
f = open("../model/Static/CombinedBaselineRF","wb")
pickle.dump(clf,f)
print("==========================================================================")



print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1]))
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]))
# create model
model = Sequential()
model.add(keras.layers.Dense(64, activation='relu',input_shape=(x_train.shape[1:])))
# model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.25))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation="softmax"))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(x_train,y_train, epochs=20, batch_size=32)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])


model_json = model.to_json()
with open("../model/Static/CombinedBaselineDNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model/Static/CombinedBaselineDNN.h5")
print("Saved model to disk")
