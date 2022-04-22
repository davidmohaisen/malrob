from shutil import copyfile
from os import listdir
from os.path import isfile, join
from os import walk
import pickle

Engines = set()


path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/packedBenign/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))


path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/packedBenign-Best/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/paddedBenign/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/benData/strippedBenign/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))


path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/packedMalware/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))


path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/packedMalware-Best/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/paddedMalware/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))

path = "/media/ahmed/HDD/ICDCS2021/Pickles/Reports/pick/malData/strippedMalware/"
files = [f for f in listdir(path) if isfile(join(path, f))]

for file in files:
    f = open(path+file,"rb")
    report = pickle.load(f)
    Engines = Engines.union(set(list(report['scans'].keys())))





setTaken = set(["MicroWorld-eScan","BitDefender","AVG","VBA32","K7GW","Rising","K7AntiVirus","ZoneAlarm","AhnLab-V3","Ad-Aware","Cyren","Acronis",
            "Zillya","TrendMicro-HouseCall","VIPRE","Comodo","ClamAV","MaxSecure","ALYac","Ikarus","CAT-QuickHeal","McAfee","Avast","Baidu","BitDefenderTheta",
            "AegisLab","Malwarebytes","FireEye","ESET-NOD32","Microsoft","TrendMicro","Avast-Mobile","Tencent","Sangfor","Fortinet","TotalDefense","MAX",
            "Yandex","Avira","Emsisoft","SUPERAntiSpyware","Antiy-AVL","Qihoo-360","Sophos","ViRobot","Bkav","Zoner","NANO-Antivirus","F-Secure","Jiangmin",
            "Gridinsoft","Cynet","Kaspersky","Arcabit","TACHYON","Symantec","McAfee-GW-Edition","Panda","Paloalto","CrowdStrike","DrWeb","SentinelOne",
            "Cybereason","CMC","GData","Kingsoft"])
setTakenAI = set(["AhnLab-V3","Cyren","Acronis","MaxSecure","Avast","BitDefenderTheta","FireEye","Avast-Mobile","Tencent","Sangfor","Fortinet",
            "MAX","SUPERAntiSpyware","Qihoo-360","Sophos","Gridinsoft","Cynet","Kaspersky","Panda","CrowdStrike","SentinelOne","Cybereason"])

# missed = Engines-Engines.intersection(setTaken)
# print(missed)



print(len(setTaken))
print(len(setTakenAI))
