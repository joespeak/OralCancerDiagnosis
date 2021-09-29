import os, random, shutil
import re
import config

def moveFile(fileDir,trainDir,testDir):
    """
    在文件层面分开测试集和训练集
    Args:
        fileDir: 图片的原始路径 例如***/firstSet/normal+ossc 有两个文件夹分别代表不同的类别
        trainDir: 训练集的路径 ***/train/   后面要有相同的类别空文件夹
        testDir: 测试集的路径 ***/test/   后面要有相同的类别空文件夹

    Returns: 无返回值

    """
    pathDir = os.listdir(fileDir)    #取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.7    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
    print(sample)

    for name in sample:
            shutil.copy(os.path.join(fileDir, name), os.path.join(trainDir, name))

    for item in pathDir:
        if item not in sample:
            shutil.copy(os.path.join(fileDir, item) , os.path.join(testDir, item))
    return

def check_train_test_construction(trainPath,testPath):
    """
    检查训练集和测试集是否有相同的图片
    Args:
        trainPath: 训练集的路径，注意要写到类别，后期会改良
        testPath: 测试集的路径，注意要写到类别，后期会改良

    Returns:无返回值

    """
    train = []
    test = []

    overlap_img = 0

    check_point = re.compile(r"(\d+).jpg$")
    label_point = re.compile(r"\\([a-z]+)$")

    label = label_point.findall(trainDir)[0]

    train_imgs = os.listdir(trainPath)
    test_imgs = os.listdir(testPath)

    # for img in train_imgs:
    #     item = check_point.findall(img)[0]
    #     train.append(item)
    #
    # for img in test_imgs:
    #     item = check_point.findall(img)[0]
    #     test.append(item)

    print(train_imgs)
    print(test_imgs)
    for img in test_imgs:
        if img in train_imgs:
            overlap_img+=1

    # for img in test:
    #     if img in train:
    #         overlap_img += 1

    print(f"{label}类别，训练集和测试集重复图片数量为{overlap_img}")

    return

def make_Peper_Out_Data(fileDir, trainDir, testDir):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径

    sample = random.sample(pathDir, 50)  # 随机选取picknumber数量的样本图片
    #print(sample)

    for name in sample:
        shutil.copy(os.path.join(fileDir, name), os.path.join(testDir, name))

    for item in pathDir:
        if item not in sample:
            shutil.copy(os.path.join(fileDir, item), os.path.join(trainDir, item))
    return

class MakeCrossValidationDataFolder():
    def __init__(self, folds, samples ,allDataFile = r"../data"):
        self.folds = folds
        self.samples = samples
        self.allDataFile = allDataFile


    def make_Peper_Out_Data(self,fileDir, trainDir, testDir):
        pathDir = os.listdir(fileDir)  # 取图片的原始路径

        if self.samples >1:
            samples = self.samples
        elif self.samples <1:
            samples = int(self.samples * len(pathDir))

        sample = random.sample(pathDir, samples)  # 随机选取picknumber数量的样本图片


        for name in sample:
            shutil.copy(os.path.join(fileDir, name), os.path.join(testDir, name))

        for item in pathDir:
            if item not in sample:
                shutil.copy(os.path.join(fileDir, item), os.path.join(trainDir, item))
        return

    def begin_split(self):
        class_list = []
        for item in glob.glob(self.allDataFile + r"/*"):
            class_list.append(item.split("/")[-1])
        print(class_list)
        for i in range(self.folds):
            if not os.path.exists(f"../{self.folds}crossValidationData/{i+1}"):
                for cls in class_list:
                    os.makedirs(f"../{self.folds}crossValidationData/{i+1}/train/{cls}")
                    os.makedirs(f"../{self.folds}crossValidationData/{i+1}/test/{cls}")

                    make_Peper_Out_Data(os.path.join(self.allDataFile, cls), f"../{self.folds}crossValidationData/{i+1}/train/{cls}",
                                        f"../{self.folds}crossValidationData/{i+1}/test/{cls}")
            else:
                break
        firstCrossDir = f"../{self.folds}crossValidationData/1"
        return firstCrossDir


if __name__ == '__main__':

    "/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/allTifData"


    fileDir = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/allTifData/normal"    #源图片文件夹路径
    trainDir = r'/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/5/train/normal'    #移动到新的文件夹路径
    testDir = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/5/test/normal"
    make_Peper_Out_Data(fileDir, trainDir, testDir)

    #myLabSetTrain = r"D:\MedicalImageProcessingPaperReading\LabSet\AllData\train\ossc"
    #myLabSetTest = r"D:\MedicalImageProcessingPaperReading\LabSet\AllData\test\ossc"

    #check_train_test_construction(trainDir, testDir)
