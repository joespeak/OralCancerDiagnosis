import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
import config

import matplotlib.pyplot as plt
from itertools import cycle
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from model import myModel
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


test_weights_path = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/EfficienNetOut/4/0.9099999666213989_epoch55normal_acc0.9399999976158142ossc_acc0.8799999952316284.pth"  # 预训练模型参数
num_class = 2  # 类别数量
gpu = "cuda:0"


# mean=[0.948078, 0.93855226, 0.9332005], var=[0.14589554, 0.17054074, 0.18254866]
def generateRoc(model, test_path):
    # 加载测试集和预训练模型参数
    #test_dir = os.path.join(data_root, 'test_images')
    #class_list = list(os.listdir(test_dir))
    #class_list.sort()

    test_dataset = dataset.ImageFolder(config.test_path, transform=config.test_transform)
    test_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=True)

    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint)
    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    pred_array = [] #predict labels
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs, _ = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = torch.max(outputs, -1)[0]  # (batchsize, nclass)
        pred_label = torch.max(outputs, -1)[1]

        score_list.extend(score_tmp.detach().cpu().numpy())
        pred_array.extend(pred_label.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    pred_array = np.array(pred_array)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = np.array(label_tensor).squeeze(-1)

    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])
    print("pred_label",pred_array.shape)

    fpr, tpr, threshold = roc_curve(label_onehot, score_array, pos_label=pred_array)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.figure(figsize=(6 * 1.2, 6))
    plt.plot(fpr, tpr, color='orange',
             lw=lw, label='AUC = %0.2f' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot')
    plt.legend(loc="lower right")

    plt.savefig(r'ROC_plot.pdf')
    print(roc_auc)
    print('ROC plot has finished!')


if __name__ == '__main__':
    # 加载模型
    net = myModel.myEfficientNet(2)
    device = torch.device(gpu)
    net = net.to(device)
    generateRoc(net, test_weights_path)