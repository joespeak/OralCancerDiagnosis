import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import re
from model.myModel import  myresNet


allPredPath = r"./DataSet/MouthCancerDataSet/LabDataSet/First Set"
print(os.listdir(allPredPath))

firstImg = os.listdir(allPredPath)[0]
labelKey = re.compile(r"^(\D+)_")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

class predDataSet(torch.utils.data.Dataset):
    def __init__(self, filesPath,transform):
        super(predDataSet, self).__init__()
        self.labelEncodeDic = {"Normal": 0, "OSCC": 1}
        self.paths = []

        for img in os.listdir(filesPath):
            self.paths.append((os.path.join(filesPath, img),self.labelEncodeDic[labelKey.findall(img)[0]]))

        self.transforms = transform
    def __getitem__(self, index):
        imgPath = self.paths[index][0]
        img = Image.open(imgPath)
        label = self.paths[index][1]

        if self.transforms:
            img = self.transforms(img)

        return imgPath, img, label

    def __len__(self):
        return len(self.paths)

if __name__ == '__main__':
    predSet = predDataSet()
    net = myresNet(2).load_state_dict(torch.load(r"./resnetState/acc0.7443181872367859_epoch9normal_acc0.2808988690376282ossc_acc0.8382688164710999.pth"))
