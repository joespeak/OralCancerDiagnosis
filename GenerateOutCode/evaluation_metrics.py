import sklearn.metrics
import os
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
import config
#from Contrast_model.ResNet_base import ResNet50

import matplotlib.pyplot as plt
from itertools import cycle
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from model import myModel

import sys
sys.path.append('..')



def get_basic_metrics(labels_true, labels_pred, class_names):
    cm = sklearn.metrics.confusion_matrix(labels_true, labels_pred, labels=range(len(class_names)))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, TP, FN, TN\


def evaluation_value_main(model,state_path,test_path, class_names):
    test_dataset = dataset.ImageFolder(test_path, transform=config.test_transform)
    test_loader = DataLoader(test_dataset, config.target_batch_size, shuffle=True)

    #model = myModel.myresNet(2).cuda()
    checkpoint = torch.load(state_path)
    model.load_state_dict(checkpoint)
    model.eval()

    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    pred_list = [] #predict labels
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        outputs, _ = model(inputs)
        # prob_tmp = torch.nn.Softmax(dim=1)(outputs) # (batchsize, nclass)
        score_tmp = torch.max(outputs, -1)[0]  # (batchsize, nclass)
        pred_label = torch.max(outputs, -1)[1]

        score_list.extend(score_tmp.detach().cpu().numpy())
        pred_list.extend(pred_label.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    pred_array = np.array(pred_list)
    # 将label转换成onehot形式
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_array = np.array(label_tensor).squeeze(-1)

    labels = label_array
    predicted = pred_array
    l = torch.from_numpy(labels)
    p = torch.from_numpy(predicted)

    PPV = sum((p==1)&(l==1)) / len(p[p == 1])
    NPV = sum((p==0)&(l==0)) / len(p[p == 0])
    # accuracy, precision, recall(Sensitivity), f1-score
    report1 = sklearn.metrics.classification_report(labels, predicted, target_names=class_names, digits=4, output_dict=True)
    report2 = sklearn.metrics.classification_report(labels, predicted, target_names=class_names, digits=4, output_dict=False)
    print("**********************************************")
    #print(sum(labels==predicted) / len(labels))
    # print(len(labels == 0))
    # print(len(labels == 1))
    # print(sum((labels==0)&(predicted==0)) / len(labels == 0))
    # print(sum((labels==1)&(predicted==1)) / len(labels == 1))

    acc_0 = sum((labels==0)&(predicted==0)) / 50
    acc_1 = sum((labels==1)&(predicted==1)) / 50

    print(sklearn.metrics.accuracy_score(labels, predicted))
    # NMI
    NMI = sklearn.metrics.normalized_mutual_info_score(labels, predicted)
    # FMI
    FMI = sklearn.metrics.fowlkes_mallows_score(labels, predicted)
    acc = report1['accuracy']
    precision = report1['macro avg']['precision']
    recall = report1['macro avg']['recall']
    f1_score = report1['macro avg']['f1-score']
    # FP, TP, FN, TN
    FP, TP, FN, TN = get_basic_metrics(labels, predicted, class_names)
    # specificity
    specificity = np.mean(TN / (TN + FP))
    # G-mean
    G_mean = np.sqrt(recall*specificity)
    the_list = [report2,
                '\n\naccuracy\tprecision\trecall\tf1-score\n',
                str(int(acc * 1e+4) / 1e+4) + '\t' + str(int(precision * 1e+4) / 1e+4) + '\t' +
                str(int(recall * 1e+4) / 1e+4) + '\t' + str(int(f1_score * 1e+4) / 1e+4) + '\n',
                'NMI:' + str(NMI) + '\n', 'FMI:' + str(FMI) + '\n'
                'sensitivity:' + str(recall) + '\n',
                'specificity:' + str(specificity) + '\n']
    #save_values(os.path.join(source_dir, 'evaluation_value.txt'), the_list, True, 'w')
    print(the_list)
    print('Evaluation value has finished!')
    res_list = [int(acc * 1e+4) / 1e+4, int(precision * 1e+4) / 1e+4, int(recall * 1e+4) / 1e+4, int(f1_score * 1e+4) / 1e+4,
                NMI, FMI, recall, specificity, PPV, NPV,acc_0,acc_1]
    return res_list


if __name__ == '__main__':
    net = myModel.myMobilieNet(2)
    device = torch.device("cuda")
    net = net.to(device)
    class_names = [str(i) for i in range(2)]

    test_path1 = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/2/test"
    test_path2 = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/3/test"
    test_path3 = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/4/test"
    test_path4 = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/5/test"
    test_path5 = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/makedTifSet/6/test"

    test_weights_path1 = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mobileBigBigOut/1/0.9599999785423279_epoch57normal_acc0.9599999785423279ossc_acc0.9599999785423279.pth"  # 预训练模型参数
    test_weights_path2 = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mobileBigBigOut/2/0.9099999666213989_epoch59normal_acc0.8399999737739563ossc_acc0.9800000190734863.pth"
    test_weights_path3 = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mobileBigBigOut/3/0.9300000071525574_epoch59normal_acc0.8999999761581421ossc_acc0.9599999785423279.pth"
    test_weights_path4 = r"/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mobileBigBigOut/4/0.9399999976158142_epoch59normal_acc0.9200000166893005ossc_acc0.9599999785423279.pth"
    test_weights_path5 = r'/home/joe/JoeFilesCollections/cap-pytorch/Contrast_model/mobileBigBigOut/5/0.9399999976158142_epoch58normal_acc0.9200000166893005ossc_acc0.9599999785423279.pth'


    res_list1 = evaluation_value_main(net, test_weights_path1, test_path1,
                          class_names=class_names)
    res_list2 = evaluation_value_main(net, test_weights_path2, test_path2,
                          class_names=class_names)
    res_list3 = evaluation_value_main(net, test_weights_path3, test_path3,
                          class_names=class_names)
    res_list4 = evaluation_value_main(net, test_weights_path4, test_path4,
                          class_names=class_names)
    res_list5 = evaluation_value_main(net, test_weights_path1, test_path5,
                          class_names=class_names)


    last_arr = np.array([res_list1,res_list2,res_list3,res_list4,res_list5])

    out_list_avg = np.mean(last_arr, 0)
    out_list_max = np.max(last_arr, 0)
    out_list_min = np.min(last_arr, 0)
    out_list_std = np.std(last_arr, 0)


    print(f"AVG---accuracy : {out_list_avg[0]}\nprecision : {out_list_avg[1]}\nrecall : {out_list_avg[2]}\nf1_score : {out_list_avg[3]}\n"
          f"NMI : {out_list_avg[4]}\nFMI : {out_list_avg[5]}\nsensitivity : {out_list_avg[6]}\n"
          f"specificity : {out_list_avg[7]}\n"
          f"PPV : {out_list_avg[8]}\nNPV : {out_list_avg[9]}\n"
          f"acc_0 : {out_list_avg[10]}\nacc_1 : {out_list_avg[11]}")
    print("*****************************************************************************************************************************")
    print(
        f"MAX---accuracy : {out_list_max[0]}\nprecision : {out_list_max[1]}\nrecall : {out_list_max[2]}\nf1_score : {out_list_max[3]}\n"
        f"NMI : {out_list_max[4]}\nFMI : {out_list_max[5]}\nsensitivity : {out_list_max[6]}\n"
        f"specificity : {out_list_max[7]}\n"
        f"PPV : {out_list_max[8]}\nNPV : {out_list_max[9]}")
    print("*****************************************************************************************************************************")
    print(
        f"MIN---accuracy : {out_list_min[0]}\nprecision : {out_list_min[1]}\nrecall : {out_list_min[2]}\nf1_score : {out_list_min[3]}\n"
        f"NMI : {out_list_min[4]}\nFMI : {out_list_min[5]}\nsensitivity : {out_list_min[6]}\n"
        f"specificity : {out_list_min[7]}\n"
        f"PPV : {out_list_min[8]}\nNPV : {out_list_min[9]}")
    print("*****************************************************************************************************************************")
    print(
        f"STD---accuracy : {out_list_std[0]}\nprecision : {out_list_std[1]}\nrecall : {out_list_std[2]}\nf1_score : {out_list_std[3]}\n"
        f"NMI : {out_list_std[4]}\nFMI : {out_list_std[5]}\nsensitivity : {out_list_std[6]}\n"
        f"specificity : {out_list_std[7]}\n"
        f"PPV : {out_list_std[8]}\nNPV : {out_list_std[9]}")


