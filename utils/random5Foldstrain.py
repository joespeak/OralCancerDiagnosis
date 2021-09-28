import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import time
from model.ResNet_base import ResNet50
import config
from trick.imbalanced import ImbalancedDatasetSampler
from tqdm import tqdm
from trick.mmd import mmd
import pandas as pd
#from trick.RandomSampleForMMD import randomSampleBatchFromSet
from model.myModel import myresNet, myRepVggNet, ArcModule, myAttentionRepVggNet, myVgg19, myMobilieNet, myEfficientNet
from torchtoolbox.tools import mixup_data
import torch.nn.functional as F
import math
import random
import os

def seed_torch(seed=2021):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def manyTimesTrain(model,firstFolds, times, modelName="mobileNet",load_model=False):
    '''
    this func aims to train Multiple partition dataset
    :param firstFolds: first partition (format shoubld be r"...../dataSetName/2" 2 means the times of which the number of partition
    :param times: How many divisions are needed in total
    :return: null
    '''

    outFiles = f"./{modelName}"+"Out"
    if not os.path.exists(outFiles):
        os.mkdir(outFiles)
        for i in range(times):
            os.mkdir(os.path.join(outFiles, f"{i+1}"))

    for i in range(times):
        originalPath = firstFolds[:-1] + str(int(firstFolds[-1]) + i)

        trainPath = os.path.join(originalPath, "train")
        testPath = os.path.join(originalPath, "test")

        trainTimes = i + 1
        print(f"***********************{trainTimes} Training Begin *************************************")
        #############################
        #begin training
        #############################
        train_dataset = dataset.ImageFolder(trainPath, transform=config.train_transform)
        # train_dataset = MouthCancerSet(config.train_path, transforms=config.A_train_transform)
        train_data_loader = DataLoader(train_dataset, config.source_batch_size,
                                       sampler=ImbalancedDatasetSampler(train_dataset))
        test_dataset = dataset.ImageFolder(testPath, transform=config.test_transform)
        # test_dataset = MouthCancerSet(config.test_path, transforms=config.A_test_transform)
        test_data_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=True)

        print(f"train共{len(train_dataset)}张")
        print(f"test共{len(test_dataset)}张")
        # define GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # define model
        # Attention !!!!!! Change your own model in here before you begin your dataset training
        net = myMobilieNet(2).to(device)

        # add arcface loss
        # arcFace_model = ArcModule(2560,config.class_num,10,0.6).to(device)

        if load_model:
            net = torch.load(config.model_path)

        cross_loss = nn.CrossEntropyLoss()  # LabelSmoothingCrossEntropy()#
        # arc_ceLoss = nn.CrossEntropyLoss()#LabelSmoothingCrossEntropy()#

        optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=5e-4)
        # optimizer = optim.SGD(net.parameters(), lr=1e-3, weight_decay=5e-4, momentum=0.9)
        # sheduler = optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

        all_acc = []
        all_acc_0 = []
        all_acc_1 = []

        for epoch in range(config.epoches):
            sum_loss = 0.
            correct = 0.
            total = 0.
            since = time.time()
            net.train()
            length = config.source_batch_size + config.target_batch_size
            dis_list = []

            train_bar = tqdm(enumerate(train_data_loader))
            for i, data in train_bar:
                inputs, labels = data
                # add cutout
                # inputs = Cutout(10, 5)(inputs)

                inputs, labels = inputs.to(device), labels.to(device)

                # print(f"normal:{torch.sum(labels==0)}")
                # print(f"ossc:{torch.sum(labels==1)}")
                optimizer.zero_grad()

                outputs, features = net(inputs)

                # arc_features = arcFace_model(features, labels)
                # outputs = cross_loss(outputs, labels)

                # last_out = F.softmax(outputs + arc_features)

                loss1 = cross_loss(outputs, labels)
                # loss2 = arc_ceLoss(arc_features, labels)

                loss = loss1  # 0.5 * loss1 + 0.5 * loss2
                sum_loss += loss

                _, pre = torch.max(outputs.data, 1)

                total += outputs.size(0)
                correct += torch.sum(pre == labels.data)
                train_acc = correct / total

                loss.backward()
                optimizer.step()
                # sheduler.step()

                iter_num = i + 1 + epoch * length
                print('[epoch:%d, iter:%d] Loss: %f | Train_acc: %f | Time: %f'
                      % (epoch + 1, iter_num, sum_loss / i, train_acc, time.time() - since))

            # start to test
            if epoch % 1 == 0:
                print("start to test:")
                with torch.no_grad():
                    correct = 0.
                    total = 0.
                    loss = 0.
                    acc_0 = 0
                    acc_1 = 0
                    total_normal_imgs = 0
                    total_ossc_imgs = 0

                    test_bar = tqdm(enumerate(test_data_loader))
                    for i, data in test_bar:
                        net.eval()
                        inputs_test, labels_test = data
                        inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                        outputs_test, _ = net(inputs_test)
                        loss += cross_loss(outputs_test, labels_test)

                        # present_max, pred = torch.max(outputs.data, 1)
                        _, pred = torch.max(outputs_test.data, 1)

                        total += labels_test.size(0)

                        total_normal_imgs += torch.sum(labels_test == 0)
                        total_ossc_imgs += torch.sum(labels_test[labels_test == 1])

                        acc_0 += torch.sum(pred[labels_test == 0] == 0)
                        acc_1 += torch.sum(pred[labels_test == 1] == 1)

                        correct += torch.sum(pred == labels_test.data)

                    test_acc = correct / total
                    print('test_acc:', test_acc, '| time', time.time() - since)
                    print(f"normal acc={acc_0 / total_normal_imgs}")
                    print(f"ossc acc={acc_1 / total_ossc_imgs}")

                    nowSavePath = os.path.join(outFiles, str(trainTimes))

                    if test_acc > 0.87 or ((epoch + 1) % 5 == 0 and epoch != 0):
                        torch.save(net.state_dict(),
                                   os.path.join(nowSavePath,
                                                f"{test_acc}_epoch{epoch}normal_acc{acc_0 / total_normal_imgs}ossc_acc{acc_1 / total_ossc_imgs}.pth"))


                    elif ((epoch + 1) > 50):
                        torch.save(net.state_dict(),
                                   os.path.join(nowSavePath,
                                                f"{test_acc}_epoch{epoch}normal_acc{acc_0 / total_normal_imgs}ossc_acc{acc_1 / total_ossc_imgs}.pth"))

                    all_acc.append(test_acc.item())

                    all_acc_0.append((acc_0 / total_normal_imgs).item())
                    all_acc_1.append((acc_1 / total_ossc_imgs).item())
                    print(all_acc)
                    print(all_acc_0)
                    print(all_acc_1)

        df_acc = pd.DataFrame({"acc": all_acc, "acc0": all_acc_0, "acc1": all_acc_1})

        df_acc.to_csv(os.path.join(nowSavePath, f"{modelName}"+"Acc"))
        #return all_acc, all_acc_0, all_acc_1

if __name__ == '__main__':
    seed_torch(2021)
    model = "mobileLarge"
    manyTimesTrain(model, config.manyTimesTrainFirstPath, times=5, modelName="mobileBigBig")
