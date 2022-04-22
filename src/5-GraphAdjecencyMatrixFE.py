import networkx as nx
# import matplotlib.pyplot as plt
import os
import sys
import pygraphviz
import numpy as np
from shutil import copyfile
import random
import pickle
#IoT malware features

pathBenign = "../graphs/benign/graphs/"
pathMalware = "../graphs/malware/graphs/"

f = open("../Pickles/filteredFilesTrainTest","rb")
FilesBenignTrain, FilesMalwareTrain, FilesBenignTest, FilesMalwareTest = pickle.load(f)

AllString = []
counter = 0

x_train = []
y_train = []
x_test = []
y_test = []

for file in FilesBenignTrain:
    print(file)
    nodes_density = []
    loc = pathBenign + file+".dot"
    g = ""
    try:
        g = nx.drawing.nx_agraph.read_dot(loc)
    except:
        print("Passed Sample")
        pass
    if g!= "" :
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.array(A)
        # print(A.shape)
        arrToAdd = np.zeros((100,100))
        # print(arrToAdd.shape)
        for j in range(min(100,len(A))):
            for k in range(min(100,len(A[j]))):
                arrToAdd[j][k] = A[j][k]
        # print(arrToAdd.shape)
        # exit()
        x_train.append(arrToAdd)
        y_train.append(0)


for file in FilesMalwareTrain:
    print(file)
    nodes_density = []
    loc = pathMalware + file+".dot"
    g = ""
    try:
        g = nx.drawing.nx_agraph.read_dot(loc)
    except:
        print("Passed Sample")
        pass
    if g!= "" :
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.array(A)
        # print(A.shape)
        arrToAdd = np.zeros((100,100))
        # print(arrToAdd.shape)
        for j in range(min(100,len(A))):
            for k in range(min(100,len(A[j]))):
                arrToAdd[j][k] = A[j][k]
        # print(arrToAdd.shape)
        # exit()
        x_train.append(arrToAdd)
        y_train.append(1)

for file in FilesBenignTest:
    print(file)
    nodes_density = []
    loc = pathBenign + file+".dot"
    g = ""
    try:
        g = nx.drawing.nx_agraph.read_dot(loc)
    except:
        print("Passed Sample")
        pass
    if g!= "" :
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.array(A)
        # print(A.shape)
        arrToAdd = np.zeros((100,100))
        # print(arrToAdd.shape)
        for j in range(min(100,len(A))):
            for k in range(min(100,len(A[j]))):
                arrToAdd[j][k] = A[j][k]
        # print(arrToAdd.shape)
        # exit()
        x_test.append(arrToAdd)
        y_test.append(0)




for file in FilesMalwareTest:
    print(file)
    nodes_density = []
    loc = pathMalware + file+".dot"
    g = ""
    try:
        g = nx.drawing.nx_agraph.read_dot(loc)
    except:
        print("Passed Sample")
        pass
    if g!= "" :
        A = nx.adjacency_matrix(g)
        A = A.todense()
        A = np.array(A)
        # print(A.shape)
        arrToAdd = np.zeros((100,100))
        # print(arrToAdd.shape)
        for j in range(min(100,len(A))):
            for k in range(min(100,len(A[j]))):
                arrToAdd[j][k] = A[j][k]
        # print(arrToAdd.shape)
        # exit()
        x_test.append(arrToAdd)
        y_test.append(1)



print(counter)


f = open("../Pickles/Graph/DataAdjacency","wb")
pickle.dump([x_train,y_train,x_test,y_test],f)
