from shutil import copyfile
from os import listdir
from os.path import isfile, join
from os import walk
import pickle

setTaken = set(["MicroWorld-eScan","BitDefender","AVG","VBA32","K7GW","Rising","K7AntiVirus","ZoneAlarm","AhnLab-V3","Ad-Aware","Cyren","Acronis",
            "Zillya","TrendMicro-HouseCall","VIPRE","Comodo","ClamAV","MaxSecure","ALYac","Ikarus","CAT-QuickHeal","McAfee","Avast","Baidu","BitDefenderTheta",
            "AegisLab","Malwarebytes","FireEye","ESET-NOD32","Microsoft","TrendMicro","Avast-Mobile","Tencent","Sangfor","Fortinet","TotalDefense","MAX",
            "Yandex","Avira","Emsisoft","SUPERAntiSpyware","Antiy-AVL","Qihoo-360","Sophos","ViRobot","Bkav","Zoner","NANO-Antivirus","F-Secure","Jiangmin",
            "Gridinsoft","Cynet","Kaspersky","Arcabit","TACHYON","Symantec","McAfee-GW-Edition","Panda","Paloalto","CrowdStrike","DrWeb","SentinelOne",
            "Cybereason","CMC","GData","Kingsoft"])

setTakenAI = set(["AhnLab-V3","Cyren","Acronis","MaxSecure","Avast","BitDefenderTheta","FireEye","Avast-Mobile","Tencent","Sangfor","Fortinet",
"MAX","SUPERAntiSpyware","Qihoo-360","Sophos","Gridinsoft","Cynet","Kaspersky","Panda","CrowdStrike","SentinelOne","Cybereason"])

setTaken = list(setTaken)
setTakenAI = list(setTakenAI)

allEngines = [0]*len(setTaken)
CorrectEngines = [0]*len(setTaken)

# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/packedBenign/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']
#
#
#
# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/packedBenign-Best/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']
#
# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/paddedBenign/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']
#
# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/strippedBenign/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']
#
#
#
# for engine in range(len(setTaken)):
#     print(setTaken[engine],allEngines[engine],CorrectEngines[engine])
# exit()

# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/packedMalware/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']

# for engine in range(len(setTaken)):
#     print(setTaken[engine],allEngines[engine],CorrectEngines[engine])
# print("\n\n\n\nAI Based")
# for engine in setTakenAI:
#     print(engine,allEngines[setTaken.index(engine)],CorrectEngines[setTaken.index(engine)])
#
# exit()
#
# path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/packedMalware-Best/"
# files = [f for f in listdir(path) if isfile(join(path, f))]
#
# for file in files:
#     f = open(path+file,"rb")
#     report = pickle.load(f)
#     for key in report['scans'].keys():
#         allEngines[setTaken.index(key)] += 1
#         CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']

# for engine in range(len(setTaken)):
#     print(setTaken[engine],allEngines[engine],CorrectEngines[engine])
# print("\n\n\n\nAI Based")
# for engine in setTakenAI:
#     print(engine,allEngines[setTaken.index(engine)],CorrectEngines[setTaken.index(engine)])
#
# exit()

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/paddedMalware/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    for key in report['scans'].keys():
        allEngines[setTaken.index(key)] += 1
        CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']

for engine in range(len(setTaken)):
    print(setTaken[engine],allEngines[engine],CorrectEngines[engine])
print("\n\n\n\nAI Based")
for engine in setTakenAI:
    print(engine,allEngines[setTaken.index(engine)],CorrectEngines[setTaken.index(engine)])

exit()

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/strippedMalware/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    for key in report['scans'].keys():
        allEngines[setTaken.index(key)] += 1
        CorrectEngines[setTaken.index(key)] += report['scans'][key]['detected']
