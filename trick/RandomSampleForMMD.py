import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import numpy as np
import time
from model.ResNet_base import ResNet50
import config
from imbalanced import ImbalancedDatasetSampler
from tqdm import tqdm
from mmd import mmd
import torchvision
import random


class randomSampleBatchFromSet():
    def __init__(self,dataset, num_samples=None, callback_get_label=None, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label
        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        self.imgs = dataset


    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[idx].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[idx][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[idx][1]
        else:
            return dataset.imgs[idx][1]

    def get_data(self,batch):
        rs_index = random.choices(self.indices,self.weights,k=batch)
        #print(rs_index)
        res_imgs = []
        res_labels = []
        for index in rs_index:
            res_imgs.append(self.imgs[index][0].unsqueeze(0))
            res_labels.append(self.imgs[index][1])

        # print(res_labels)
        res_imgs = torch.cat(res_imgs,dim=0)
        res_labels = torch.tensor(res_labels)
        # print(f"normal:{torch.sum(res_labels)}")
        # print(res_imgs.shape)
        # print(type(res_imgs))
        # print(type(res_labels))
        return res_imgs,res_labels
        # return (self.indices[i] for i in torch.multinomial(
        #     self.weights, self.num_samples, replacement=True))


if __name__ == '__main__':

    test_dataset = dataset.ImageFolder(config.test_path, transform=config.test_transform)

    rs = randomSampleBatchFromSet(test_dataset,60)
    print(rs.get_data())
    #print(test_dataset[0])
    print(len(test_dataset))
