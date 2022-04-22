import os, pickle


def main():
    f = open("../Pickles/filteredFilesTrainTest","rb")
    _, _, FilesBenignTest, FilesMalwareTest = pickle.load(f)

    malPath = "/media/ahmed/HDD/MalwareTemporalRobustness/Data/ELF/filtered/Malware/"

    malwareList = set(os.listdir(malPath)).intersection(set(FilesMalwareTest))
    upxExecPath = "/home/ahmed/Documents/Projects/RobustnessAnalysis/UPX/upx"

    newMalPath = "../data/malData/"
    if not os.path.isdir(newMalPath):
        os.mkdir(newMalPath)


    for mal in malwareList:
        newMalPath = "../data/malData/Original/"
        if not os.path.isdir(newMalPath):
            os.mkdir(newMalPath)
        os.system("cp "+malPath+mal+" "+newMalPath+mal)


    malPath = "/media/ahmed/HDD/MalwareTemporalRobustness/Data/ELF/filtered/Benign/"

    malwareList = set(os.listdir(malPath)).intersection(set(FilesBenignTest))
    upxExecPath = "/home/ahmed/Documents/Projects/RobustnessAnalysis/UPX/upx"

    newMalPath = "../data/benData/"
    if not os.path.isdir(newMalPath):
        os.mkdir(newMalPath)


    for mal in malwareList:
        newMalPath = "../data/benData/Original/"
        if not os.path.isdir(newMalPath):
            os.mkdir(newMalPath)
        os.system("cp "+malPath+mal+" "+newMalPath+mal)




if __name__=="__main__":
    main()
