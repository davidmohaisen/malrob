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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from PIL import Image
from art.classifiers import KerasClassifier
import sys
sys.path.append(".")
from WhiteBoxAttack import CarliniLInfMethod

label = "Baseline"
size = (64,64)
f = open("../Pickles/Images/"+label+str(size),"rb")
_, _, x_test, y_test = pickle.load(f)

print(x_test.shape)
print(y_test.shape)

# Input image dimensions.

# Normalize data.
x_test = x_test.astype('float32') / 255



print("==========================================================================")
print("LR")
print("==========================================================================")
x_test_new_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
f = open("../model/Image/BaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test_new_, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
x_test_new_ = np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))
f = open("../model/Image/BaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_new_, y_test)
print("The RF score is:",score)

print("==========================================================================")


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
# Convert class vectors to binary class matrices.
y_test_n = keras.utils.to_categorical(y_test, 2)
x_test_new = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

json_file = open('../model/Image/BaselineDNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Image/BaselineDNN.h5")

model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test_n, verbose=1)
print('Test accuracy:', scores[1])


# x_test = x_test.astype('float32') * 255
compressed_x = []
for i in range(len(x_test)):
    pic = Image.fromarray(x_test[i])
    pic = pic.resize((64,32));
    pix = np.array(pic)
    # pic.show()

    layer = Image.new('I', (64,64), 128)
    layer.paste(pic)
    # layer.show()
    pix = np.array(layer)
    # print(pix.shape)
    compressed_x.append(pix)

compressed_x = np.asarray(compressed_x)
print(compressed_x.shape)




classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)


predictions = classifier.predict(x_test_new)
# print(predictions)
# print(y_test)
# exit()
accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


attack = CarliniLInfMethod(classifier=classifier,max_iter=100, eps=(256/256),learning_rate=(16/256))
x_train_adv = attack.generate(x=x_test_new)

f = open("../Pickles/Images/ImageBinaryInjection","wb")
pickle.dump(x_train_adv,f)

for i in range(len(x_test_new[0])):
    print(x_test_new[0])
    print(x_train_adv[0])

print("==========================================================================")
print("LR")
print("==========================================================================")
x_test_new_ = np.reshape(x_train_adv,(x_train_adv.shape[0],x_train_adv.shape[1]*x_train_adv.shape[2]))
f = open("../model/Image/BaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test_new_, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
x_test_new_ = np.reshape(x_train_adv,(x_train_adv.shape[0],x_train_adv.shape[1]*x_train_adv.shape[2]))
f = open("../model/Image/BaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_new_, y_test)
print("The RF score is:",score)

print("==========================================================================")


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
# Convert class vectors to binary class matrices.
y_test_n = keras.utils.to_categorical(y_test, 2)
x_test_new = np.reshape(x_train_adv,(x_train_adv.shape[0],x_train_adv.shape[1],x_train_adv.shape[2],1))

json_file = open('../model/Image/BaselineDNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Image/BaselineDNN.h5")

model.compile(loss='categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test_n, verbose=1)
print('Test accuracy:', scores[1])
