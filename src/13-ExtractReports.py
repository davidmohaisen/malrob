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


dic = {"Avast":"E --- 1","Qihoo-360":"E --- 2","Cynet":"E --- 3","FireEye":"E --- 4","SentinelOne":"E --- 5","Kaspersky":"E --- 6","Acronis":"E --- 7","Sophos":"E --- 8","MAX":"E --- 9","AhnLab-V3":"E --- 10","Fortinet":"E --- 11","Avast-Mobile":"E --- 12","SUPERAntiSpyware":"E --- 13","Gridinsoft":"E --- 14","MaxSecure":"E --- 15","BitDefenderTheta":"E --- 16","Cyren":"E --- 17","Tencent":"E --- 18","Sangfor":"E --- 19","Panda":"E --- 20","CrowdStrike":"E --- 21","Comodo":"E --- 22","ESET-NOD32":"E --- 23","Zillya":"E --- 24","AegisLab":"E --- 25","Symantec":"E --- 26","AVG":"E --- 27","ClamAV":"E --- 28","Malwarebytes":"E --- 29","ViRobot":"E --- 30","K7AntiVirus":"E --- 31","DrWeb":"E --- 32","F-Secure":"E --- 33","Avira":"E --- 34","GData":"E --- 35","McAfee":"E --- 36","TotalDefense":"E --- 37","TACHYON":"E --- 38","McAfee-GW-Edition":"E --- 39","Emsisoft":"E --- 40","VIPRE":"E --- 41","K7GW":"E --- 42","Bkav":"E --- 43","Antiy-AVL":"E --- 44","Zoner":"E --- 45","TrendMicro-HouseCall":"E --- 46","Arcabit":"E --- 47","VBA32":"E --- 48","Ad-Aware":"E --- 49","ALYac":"E --- 50","Jiangmin":"E --- 51","Ikarus":"E --- 52","BitDefender":"E --- 53","Yandex":"E --- 54","NANO-Antivirus":"E --- 55","Kingsoft":"E --- 56","TrendMicro":"E --- 57","CAT-QuickHeal":"E --- 58","Rising":"E --- 59","Microsoft":"E --- 60","CMC":"E --- 61","MicroWorld-eScan":"E --- 62","ZoneAlarm":"E --- 63","Baidu":"E --- 64","Cybereason":"NotUsed","Paloalto":"NotUsed"}


setTakenAI = set(["AhnLab-V3","Cyren","Acronis","MaxSecure","Avast","BitDefenderTheta","FireEye","Avast-Mobile","Tencent","Sangfor","Fortinet",
"MAX","SUPERAntiSpyware","Qihoo-360","Sophos","Gridinsoft","Cynet","Kaspersky","Panda","CrowdStrike","SentinelOne","Cybereason"])

setTaken = list(setTaken)
setTakenAI = list(setTakenAI)

Data = [""]*len(setTaken)
Header = "Engine,AI,Original Benign %, Packed Benign %, Packed* Benign %, Stripped Benign %, Padded Benign %,Original Malware %, Packed Malware %, Packed* Malware %, Stripped Malware %, Padded Malware %,\n"
for engineInd in range(len(setTaken)):
    Data[engineInd]+= dic[setTaken[engineInd]]+","
    if setTaken[engineInd] in setTakenAI:
        Data[engineInd]+= "\\tick,"
    else:
        Data[engineInd]+= "\\xmark,"

    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/benData/Original/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct += not report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"



    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/benData/packedBenign/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct += not report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"


    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/benData/packedBenign-Best/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct += not report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"

    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/benData/strippedBenign/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct += not report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"


    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/benData/paddedBenign/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct += not report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"

    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/malData/Original/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct +=  report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"


    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/malData/packedMalware/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct +=  report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"


    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/malData/packedMalware-Best/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct +=  report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"

    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/malData/strippedMalware/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct +=  report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"


    path = "/media/ahmed/HDD/ACSAC2021/Pickles/Reports/pick/malData/paddedMalware/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    All = 0
    Correct = 0
    for file in files:
        f = open(path+file,"rb")
        report = pickle.load(f)
        for key in report['scans'].keys():
            if key == setTaken[engineInd]:
                All += 1
                Correct +=  report['scans'][key]['detected']
    try:
        Data[engineInd]+= str(round(100*Correct/All,2))+","
    except:
        Data[engineInd]+= "---,"
f = open("../EnginesAnalysis/Industry_Scan.csv","w")
f.write(Header)
for line in Data:
    f.write(line+"\n")

f.close()
