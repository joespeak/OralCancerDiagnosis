import os
import torchvision.transforms as transforms
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2

class_num = 2

# train_batch_size
source_batch_size = 1
# test_batch_size
target_batch_size = 1

learning_rate = 1e-5

#the begin path of using crossvalidation(run random5Foldstrain.py)
manyTimesTrainFirstPath = r"/MouthCancerDataSet/LabDataSet/makedTifSet/2"
#the train path for a single data to training(run train_base.py)
train_path = r'/MouthCancerDataSet/LabDataSet/makedTifSet/5/train'
#the test path for a single data to training(run train_base.py)
test_path = r'/MouthCancerDataSet/LabDataSet/makedTifSet/5/test'

#model_path = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mmdState/PureMineacc0.6666666269302368_epoch29normal_acc0.6338028311729431ossc_acc0.7115384340286255.pth"

epoches = 60

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Grayscale(),
])



test_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Grayscale(),
])

A_train_transform = A.Compose([A.HorizontalFlip(p=0.5),
                                            A.HorizontalFlip(p=0.5),
                                            A.VerticalFlip(p=0.5),
                                            #A.RandomBrightness(limit=0.2, p=0.5),
                                            #A.RandomContrast(limit=0.2, p=0.5),
                                            # A.OneOf([
                                            #     A.MotionBlur(blur_limit=3),
                                            #     A.MedianBlur(blur_limit=3),
                                            #     A.GaussianBlur(blur_limit=3),
                                            #     A.GaussNoise(var_limit=(3.0, 30.0)),
                                            # ], p=0.5),
                                            # A.OneOf([
                                            #     A.OpticalDistortion(distort_limit=1.0),
                                            #     A.GridDistortion(num_steps=5, distort_limit=1.),
                                            #     A.ElasticTransform(alpha=3),
                                            # ], p=0.7),

                                            A.Resize(224, 224),
                                            # A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375),
                                            #          num_holes=1, p=0.7),
                                            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ToTensorV2()])
A_test_transform = A.Compose([
    A.Resize(224, 224),
    # A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375),
    #          num_holes=1, p=0.7),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ToTensorV2()
])











