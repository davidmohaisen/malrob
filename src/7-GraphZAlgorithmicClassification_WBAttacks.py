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
import networkx as nx
f = open("../Pickles/Graph/DataAlgorithmic","rb")
x_train,y_train,x_test,y_test = pickle.load(f)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)



print("==========================================================================")
print("LR")
print("==========================================================================")
f = open("../model/Graph/AlgorithmicBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test, y_test)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Graph/AlgorithmicBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test, y_test)
print("The RF score is:",score)

print("==========================================================================")


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test_new = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

json_file = open('../model/Graph/AlgorithmicBaselineCNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Graph/AlgorithmicBaselineCNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test, verbose=1)
print('Test accuracy:', scores[1])




f = open("../Pickles/Graph/AdjacencyAdversarial","rb")
x_adv = pickle.load(f)
x_adv = np.where(x_adv>=0.5, 1, x_adv)
x_adv = np.where(x_adv<0.5, 0, x_adv)
x_adv = np.reshape(x_adv,(x_adv.shape[0],x_adv.shape[1],x_adv.shape[2]))


f = open("../Pickles/Graph/DataAdjacency","rb")
_,_,_,y_test = pickle.load(f)
x_test_new = []
y_test_new = []

for graphAdj in range(len(x_adv)):
    g = nx.from_numpy_matrix(x_adv[graphAdj])
    g.remove_nodes_from(list(nx.isolates(g)))

    #### Start from here ####
    node_cnt = len(list(nx.nodes(g)))
    edge_cnt = len(list(nx.edges(g)))
    shortest_path = []
    closeness = []
    diameter = 0
    radius = 0
    current_flow_closeness = ""

    shortest_path = nx.shortest_path(g)
    closeness = nx.algorithms.centrality.closeness_centrality(g)
    # shortest_betweenness = nx.algorithms.centrality.betweenness_centrality(g)
    degree_centrality = nx.algorithms.centrality.degree_centrality(g)
    density = nx.density(g)

    sp_length = []
    for i in shortest_path:
        sp_length.append(shortest_path[i])
    shortestPathsArray = []
    for i in range(len(sp_length)):
        for x in sp_length[i] :
            if (len(sp_length[i][x])-1)==0 :
                continue
            shortestPathsArray.append((len(sp_length[i][x])-1))

    if (len(shortestPathsArray))== 0 :
        continue

    maxShortestPath = np.max(shortestPathsArray)
    minShortestPath = np.min(shortestPathsArray)
    meanShortestPath = np.mean(shortestPathsArray)
    medianShortestPath = np.median(shortestPathsArray)
    stdShortestPath = np.std(shortestPathsArray)
    closeness_list = list(closeness.values())
    # betweenness_list = list(shortest_betweenness.values())
    degree_list = list(degree_centrality.values())
    out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
    # out = [np.max(degree_list),np.min(degree_list),np.mean(degree_list),np.median(degree_list),np.std(degree_list),np.max(betweenness_list),np.min(betweenness_list),np.mean(betweenness_list),np.median(betweenness_list),np.std(betweenness_list),np.max(closeness_list),np.min(closeness_list),np.mean(closeness_list),np.median(closeness_list),np.std(closeness_list),maxShortestPath,minShortestPath,meanShortestPath,medianShortestPath,stdShortestPath,node_cnt,edge_cnt,density]
    x_test_new.append(out)
    y_test_new.append(y_test[graphAdj])
x_test_new = np.asarray(x_test_new)
y_test_new = np.asarray(y_test_new)


print("==========================================================================")
print("LR")
print("==========================================================================")
f = open("../model/Graph/AlgorithmicBaselineLR","rb")
clf = pickle.load(f)

score = clf.score(x_test_new, y_test_new)
print("The LR score is:",score)

print("==========================================================================")
print("RF")
print("==========================================================================")
f = open("../model/Graph/AlgorithmicBaselineRF","rb")
clf = pickle.load(f)
score = clf.score(x_test_new, y_test_new)
print("The RF score is:",score)

print("==========================================================================")


print("==========================================================================")
print("Deep Learning")
print("==========================================================================")
x_test_new = np.reshape(x_test_new,(x_test_new.shape[0],x_test_new.shape[1],1))

json_file = open('../model/Graph/AlgorithmicBaselineCNN.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("../model/Graph/AlgorithmicBaselineCNN.h5")

model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

scores = model.evaluate(x_test_new, y_test_new, verbose=1)
print('Test accuracy:', scores[1])
