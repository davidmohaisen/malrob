from shutil import copyfile
from os import listdir
from os.path import isfile, join
from os import walk
import pickle

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/Random/"


files = [f for f in listdir(path) if isfile(join(path, f))]
# print(files)

SearchEngines = set()

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    print(report)
    exit()
    SearchEngines = SearchEngines.union(set(report['scans'].keys()))

SearchEngines = list(SearchEngines)

print(len(SearchEngines))
print(SearchEngines)


# BitDefender: -
# MicroWorld-eScan: -
# AVG : -
# VBA32: -
# K7GW: -
# Rising: -
# K7AntiVirus: -
# ZoneAlarm: -
# AhnLab-V3: AI
# Ad-Aware: -
# Cyren: AI
# Acronis: AI
# Zillya: -
# TrendMicro-HouseCall: -
# VIPRE: -
# Comodo: -
# ClamAV: -
# MaxSecure: AI
# ALYac: -
# Ikarus: -
# CAT-QuickHeal: -
# McAfee: -
# Avast: AI
# Baidu: -
# BitDefenderTheta: AI
# AegisLab: -
# Malwarebytes: -
# FireEye: AI
# ESET-NOD32: -
# Microsoft: -
# TrendMicro: -
# Avast-Mobile: AI
# Tencent: AI
# Sangfor: AI
# Fortinet: AI
# TotalDefense: -
# MAX: AI
# Yandex: -
# Avira: -
# Emsisoft: -
# SUPERAntiSpyware: AI
# Antiy-AVL: -
# Qihoo-360: AI
# Sophos: AI
# ViRobot: -
# Bkav: -
# Zoner: -
# NANO-Antivirus: -
# F-Secure: -
# Jiangmin: -
# Gridinsoft: AI
# Cynet: AI
# Kaspersky: AI
# Arcabit: -
# TACHYON: -
# Symantec: -
# McAfee-GW-Edition: -
# Panda: AI
