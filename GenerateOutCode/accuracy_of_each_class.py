import sklearn.metrics
import numpy as np
import os

import sys
sys.path.append('..')



def get_acc_of_each_class(labels_true, labels_pred, class_names, source_dir):
    cm = sklearn.metrics.confusion_matrix(labels_true, labels_pred)
    diag_list = np.diag(cm)
    each_class_number = np.sum(cm, axis=1)
    acc_list = []
    for i in range(len(class_names)):
        class_acc = diag_list[i] / each_class_number[i]
        acc_list.append(class_names[i] + '\t' + str(class_acc) + '\n')
    save_values(os.path.join(source_dir, 'each_class_accuracy.txt'), acc_list, True, 'w')
    print('Acc of each class has finished!')


def acc_of_each_class_main(labels_true_path, labels_pred_path, class_names, source_dir):
    labels_true = load_values(labels_true_path)
    labels_pred = load_values(labels_pred_path)
    labels_true = np.array(labels_true, dtype=np.int)
    labels_pred = np.array(labels_pred, dtype=np.int)
    get_acc_of_each_class(labels_true, labels_pred, class_names, source_dir)


if __name__ == '__main__':
    source_dir = r'D:\mengxinagjie\code\cser\mengxiangjie_code\contrast_experience\KNN\two'
    labels_true_path = os.path.join(source_dir, 'labels_true-2.npy')
    labels_pred_path = os.path.join(source_dir, 'labels_pred-2.npy')
    class_names = [str(i) for i in range(2)]
    acc_of_each_class_main(labels_true_path=labels_true_path,
                           labels_pred_path=labels_pred_path,
                           class_names=class_names,
                           source_dir=source_dir)












