import random

import sklearn.decomposition as dp
import pandas as pd
import numpy as np
import sklearn.svm
from PIL import Image
import os
from sklearn.utils import shuffle
import operator
import sklearn.metrics
from model.ELM import RELM_HiddenLayer, HiddenLayer

def get_basic_metrics(labels_true, labels_pred, class_names):
    cm = sklearn.metrics.confusion_matrix(labels_true, labels_pred, labels=range(len(class_names)))
    cm = cm.astype(np.float32)
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    return FP, TP, FN, TN\

def getLastOut(preds, labels, class_names):
    p = preds
    l = labels
    PPV = sum((p == 1) & (l == 1)) / len(p[p == 1])
    NPV = sum((p == 0) & (l == 0)) / len(p[p == 0])

    # accuracy, precision, recall(Sensitivity), f1-score
    report1 = sklearn.metrics.classification_report(labels, preds, target_names=class_names, digits=4,
                                                    output_dict=True)
    report2 = sklearn.metrics.classification_report(labels, preds, target_names=class_names, digits=4,
                                                    output_dict=False)
    print("**********************************************")
    acc_0 = sum((labels == 0) & (preds == 0)) / 50
    acc_1 = sum((labels == 1) & (preds == 1)) / 50

    # NMI
    NMI = sklearn.metrics.normalized_mutual_info_score(labels, preds)
    # FMI
    FMI = sklearn.metrics.fowlkes_mallows_score(labels, preds)
    acc = report1['accuracy']
    precision = report1['macro avg']['precision']
    recall = report1['macro avg']['recall']
    f1_score = report1['macro avg']['f1-score']
    # FP, TP, FN, TN
    FP, TP, FN, TN = get_basic_metrics(labels, preds, class_names)
    # specificity
    specificity = np.mean(TN / (TN + FP))
    # G-mean
    G_mean = np.sqrt(recall * specificity)
    the_list = [report2,
                '\n\naccuracy\tprecision\trecall\tf1-score\n',
                str(int(acc * 1e+4) / 1e+4) + '\t' + str(int(precision * 1e+4) / 1e+4) + '\t' +
                str(int(recall * 1e+4) / 1e+4) + '\t' + str(int(f1_score * 1e+4) / 1e+4) + '\n',
                'NMI:' + str(NMI) + '\n', 'FMI:' + str(FMI) + '\n'
                                                              'sensitivity:' + str(recall) + '\n',
                'specificity:' + str(specificity) + '\n']
    # save_values(os.path.join(source_dir, 'evaluation_value.txt'), the_list, True, 'w')
    print(the_list)
    print('Evaluation value has finished!')
    res_list = [int(acc * 1e+4) / 1e+4, int(precision * 1e+4) / 1e+4, int(recall * 1e+4) / 1e+4,
                int(f1_score * 1e+4) / 1e+4,
                NMI, FMI, recall, specificity, PPV, NPV, acc_0, acc_1]


    return res_list



def generate_dataFromImg(originalPath, classNames, samples,mode=0):
    """
    generate data for machine-learning model train and test
    :param originalPath: all class folders' parent folder
    :param classNames: all class folders names -> String List
    :param mode: choose the format of generated data
                mode =0 means generate tuple data eg:(img,label)
                mode = 1 mens generate two lists data eg [img],[label]
    :return: generated data
    """
    encodeLabels = [x for x in range(len(classNames))]

    if mode == 1:
        imgs = []
        labels = []
        classDivingLine = 0

        for i in range(len(classNames)):
            cls = classNames[i]
            cls_foder = os.path.join(originalPath, cls)
            imgs_list = os.listdir(cls_foder)

            for img in imgs_list:
                img_path = os.path.join(cls_foder, img)
                iiii = Image.open(img_path).convert("L")
                iiii = iiii.resize(128, 128)
                imgData = np.array(iiii)
                imgData = imgData.flatten()
                imgs.append(imgData)
                labels.append(encodeLabels[i])

            classDivingLine = (len(labels))

        normalData = imgs[:classDivingLine]
        normalLabels = labels[:classDivingLine]
        osscData = imgs[classDivingLine:]
        osscLabels = labels[classDivingLine:]

        testNormal = random.sample(normalData, samples)
        for img in normalData:
            if img in testNormal:
                normalData.remove(img)
        trainNormal = normalData

        testOssc = random.sample(osscData, samples)
        for img in osscData:
            if img in testOssc:
                osscData.remove(img)
        trainOssc = osscData

    elif mode == 0:
        imgs = []
        normalImgs = []
        osscImgs = []

        for i in range(len(classNames)):
            cls = classNames[i]
            cls_foder = os.path.join(originalPath, cls)
            imgs_list = os.listdir(cls_foder)

            for img in imgs_list:
                img_path = os.path.join(cls_foder, img)

                iiii = Image.open(img_path).convert("L")
                iiii = iiii.resize((224, 224))
                imgData = np.array(iiii)
                imgData = np.array(iiii)
                imgData = imgData.flatten()
                imgs.append((imgData, encodeLabels[i]))

        for img in imgs:
            if img[1] == 0:
                normalImgs.append(img)
            elif img[1] == 1:
                osscImgs.append(img)

        trainNormal = []
        testNormal = random.sample(normalImgs, samples)
        for img in normalImgs:
            for iii in testNormal:
                if (img[0] == iii[0]).all() and img[1] == iii[1]:
                    pass
                else:
                    trainNormal.append(img)
                    break


        trainOssc = []
        testOssc = random.sample(osscImgs, samples)
        for img in osscImgs:
            for iii in testOssc:
                if (img[0] == iii[0]).all() and img[1] == iii[1]:
                    pass
                else:
                    trainOssc.append(img)
                    break

        # print(len(trainNormal))
        # print(len(trainOssc))
        # print(trainNormal[0])
        trainNormal.extend(trainOssc)
        print(f"train : {len(trainNormal)}")
        testNormal.extend(testOssc)
        print(f"test : {len(testNormal)}")

        return trainNormal, testNormal

def PCAandELM(train, test):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for item in train:
        train_x.append(item[0])
        train_y.append(item[1])

    for item in test:
        test_x.append(item[0])
        test_y.append(item[1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    all_img = np.concatenate((train_x, test_x), axis=0)
    print(all_img.shape)
    pca = dp.PCA(0.95)
    print("PCA is extracting")
    pca_imgs = pca.fit_transform(all_img)
    train_pca = pca_imgs[:len(train_x)]
    test_pca = pca_imgs[len(train_x):]

    print("ELM is traing")
    model = HiddenLayer(train_pca,50)
    model.classifisor_train(train_y)
    print("ELM is predicting")
    preds = model.classifisor_test(test_pca)
    preds = np.array(preds)

    return preds, test_y

def PCAandSVM(train, test):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for item in train:
        train_x.append(item[0])
        train_y.append(item[1])

    for item in test:
        test_x.append(item[0])
        test_y.append(item[1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    all_img = np.concatenate((train_x, test_x), axis=0)
    print(all_img.shape)
    pca = dp.PCA(0.95)
    print("PCA is extracting")
    pca_imgs = pca.fit_transform(all_img)
    train_pca = pca_imgs[:len(train_x)]
    test_pca = pca_imgs[len(train_x):]

    print("SVM is traing")
    model = sklearn.svm.SVC()
    model.fit(train_pca, train_y)

    print("SVM is predicting")
    preds = model.predict(test_pca)

    return preds, test_y


if __name__ == '__main__':
    class_names = [str(i) for i in range(2)]
    oPath = r"/home/joe/JoeFilesCollections/cap-pytorch/DataSet/MouthCancerDataSet/LabDataSet/allTifData"

    out_list = []
    for i in range(5):
        train_data, test_data = generate_dataFromImg(oPath, ["normal", "ossc"], 50)
        preds, labels = PCAandELM(train_data, test_data)
        res_list = getLastOut(preds, labels, class_names)
        out_list.append(res_list)

    last_arr = np.array(out_list)

    out_list_avg = np.mean(last_arr, 0)
    out_list_max = np.max(last_arr, 0)
    out_list_min = np.min(last_arr, 0)
    out_list_std = np.std(last_arr, 0)

    print(
        f"AVG---accuracy : {out_list_avg[0]}\nprecision : {out_list_avg[1]}\nrecall : {out_list_avg[2]}\nf1_score : {out_list_avg[3]}\n"
        f"NMI : {out_list_avg[4]}\nFMI : {out_list_avg[5]}\nsensitivity : {out_list_avg[6]}\n"
        f"specificity : {out_list_avg[7]}\n"
        f"PPV : {out_list_avg[8]}\nNPV : {out_list_avg[9]}\n"
        f"acc_0 : {out_list_avg[10]}\nacc_1 : {out_list_avg[11]}")
    print(
        "*****************************************************************************************************************************")
    print(
        f"MAX---accuracy : {out_list_max[0]}\nprecision : {out_list_max[1]}\nrecall : {out_list_max[2]}\nf1_score : {out_list_max[3]}\n"
        f"NMI : {out_list_max[4]}\nFMI : {out_list_max[5]}\nsensitivity : {out_list_max[6]}\n"
        f"specificity : {out_list_max[7]}\n"
        f"PPV : {out_list_max[8]}\nNPV : {out_list_max[9]}")
    print(
        "*****************************************************************************************************************************")
    print(
        f"MIN---accuracy : {out_list_min[0]}\nprecision : {out_list_min[1]}\nrecall : {out_list_min[2]}\nf1_score : {out_list_min[3]}\n"
        f"NMI : {out_list_min[4]}\nFMI : {out_list_min[5]}\nsensitivity : {out_list_min[6]}\n"
        f"specificity : {out_list_min[7]}\n"
        f"PPV : {out_list_min[8]}\nNPV : {out_list_min[9]}")
    print(
        "*****************************************************************************************************************************")
    print(
        f"STD---accuracy : {out_list_std[0]}\nprecision : {out_list_std[1]}\nrecall : {out_list_std[2]}\nf1_score : {out_list_std[3]}\n"
        f"NMI : {out_list_std[4]}\nFMI : {out_list_std[5]}\nsensitivity : {out_list_std[6]}\n"
        f"specificity : {out_list_std[7]}\n"
        f"PPV : {out_list_std[8]}\nNPV : {out_list_std[9]}")