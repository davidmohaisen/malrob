import requests
import json
import os
import subprocess
import cv2
import numpy
import pickle
width = 64
height = 160
#Benign
path = "/media/ahmed/HDD/ICDCS2021/data/benData/packedBenign/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/packedBenign/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
#Malware
path = "/media/ahmed/HDD/ICDCS2021/data/benData/packedBenign-Best/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/packedBenign-Best/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
path = "/media/ahmed/HDD/ICDCS2021/data/benData/paddedBenign/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/paddedBenign/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()


path = "/media/ahmed/HDD/ICDCS2021/data/benData/strippedBenign/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/strippedBenign/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)https://www.cs.ucf.edu/cs_timeclock/
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()



path = "/media/ahmed/HDD/ICDCS2021/data/malData/packedMalware/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/packedMalware/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
#Malware
path = "/media/ahmed/HDD/ICDCS2021/data/malData/packedMalware-Best/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/packedMalware-Best/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
path = "/media/ahmed/HDD/ICDCS2021/data/malData/paddedMalware/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/paddedMalware/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()


path = "/media/ahmed/HDD/ICDCS2021/data/malData/strippedMalware/"
pathG = "/media/ahmed/HDD/ICDCS2021/images/Manipulated/strippedMalware/"
for sample in os.listdir(path):
    file_to_scan = path+sample
    if not os.path.exists(pathG+sample):
        my_list = []
        f = open(file_to_scan, "rb")
        try:
            byte = f.read(1)
            currentVector = []
            while True :
                currentVector.append(ord(byte))
                if len(currentVector)==width :
                    my_list.append(currentVector)
                    currentVector = []
                byte = f.read(1)
        except:
            if len(currentVector)!= 0:
                toAppend = 128
                for i in range(width-len(currentVector)) :
                    currentVector.append(toAppend)
                    toAppend = 0
                my_list.append(currentVector)
            f.close()

        img = cv2.resize(numpy.asarray(my_list, dtype=numpy.uint8), dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        f = open(pathG+sample,"wb")
        pickle.dump(img,f)
        f.close()
