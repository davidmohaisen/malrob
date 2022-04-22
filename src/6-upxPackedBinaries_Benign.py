import os, pickle

def packer(mal, malPath, upxExecPath, degree):
    os.system("chmod +x "+malPath+mal)
    os.system(upxExecPath+" "+degree+malPath+mal)
    # os.system("objcopy --strip-all "+malPath+mal+" "+outPath+mal)
    # os.system("objcopy --strip-debug "+outPath+mal)
    # os.system("objcopy --remove-section=.* "+outPath+mal)
    print(mal, "Packed!!!")
    # objcopy --strip-debug VirusShare_0e05896b109cab25b5dd680cbf416015
    # objcopy --remove-section=.* VirusShare_0e05896b109cab25b5dd680cbf416015

def stripper(mal, malPath):
    os.system("chmod +x "+malPath+mal)
    os.system("objcopy --strip-unneeded --strip-debug "+malPath+mal)
    # os.system("objcopy --remove-section=.* "+outPath+mal)
    print(mal, "Stripped!!!")

def main():
    f = open("../Pickles/filteredFilesTrainTest","rb")
    _, _, FilesBenginTest, _ = pickle.load(f)

    malPath = "/media/ahmed/HDD/MalwareTemporalRobustness/Data/ELF/filtered/Benign/"

    malwareList = set(os.listdir(malPath)).intersection(set(FilesBenginTest))
    upxExecPath = "/home/ahmed/Documents/Projects/RobustnessAnalysis/UPX/upx"

    newMalPath = "../data/benData/"
    if not os.path.isdir(newMalPath):
        os.mkdir(newMalPath)


    for mal in malwareList:
        newMalPath = "../data/benData/packedBenign/"
        if not os.path.isdir(newMalPath):
            os.mkdir(newMalPath)
        os.system("cp "+malPath+mal+" "+newMalPath+mal)
        degree = ""
        packer(mal, newMalPath, upxExecPath, degree)

    for mal in malwareList:
        newMalPath = "../data/benData/strippedBenign/"
        if not os.path.isdir(newMalPath):
            os.mkdir(newMalPath)
        os.system("cp "+malPath+mal+" "+newMalPath+mal)
        stripper(mal, newMalPath)

    for mal in malwareList:
        newMalPath = "../data/benData/packedBenign-Best/"
        if not os.path.isdir(newMalPath):
            os.mkdir(newMalPath)
        os.system("cp "+malPath+mal+" "+newMalPath+mal)
        degree = "--best "
        packer(mal, newMalPath, upxExecPath, degree)

if __name__=="__main__":
    main()
