import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  #数据集的分割函数
from sklearn.preprocessing import StandardScaler      #数据预处理
from sklearn import metrics
import pandas as pd
from sklearn.utils import shuffle

class RELM_HiddenLayer:

    """
        正则化的极限学习机
        :param x: 初始化学习机时的训练集属性X
        :param num: 学习机隐层节点数
        :param C: 正则化系数的倒数
    """

    def __init__(self, x, num, C=10):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()
        # 权重w
        self.w = rnd.uniform(-1, 1, (columns, num))
        # 偏置b
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        self.H0 = np.matrix(self.softplus(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I
        #.T:转置矩阵,.H:共轭转置,.I:逆矩阵

    @staticmethod
    def sigmoid(x):
        """
            激活函数sigmoid
            :param x: 训练集中的X
            :return: 激活值
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        """
            激活函数 softplus
            :param x: 训练集中的X
            :return: 激活值
        """
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        """
            激活函数tanh
            :param x: 训练集中的X
            :return: 激活值
        """
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # 分类问题 训练
    def classifisor_train(self, T):
        """
            初始化了学习机后需要传入对应标签T
            :param T: 对应属性X的标签T
            :return: 隐层输出权值beta
        """
        if len(T.shape) > 1:
            pass
        else:
            self.en_one = OneHotEncoder()
            T = self.en_one.fit_transform(T.reshape(-1, 1)).toarray()
            pass
        all_m = np.dot(self.P, self.H0.H)
        self.beta = np.dot(all_m, T)
        return self.beta

    # 分类问题 测试
    def classifisor_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = test_x.shape[0]
        h = self.softplus(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result =np.argmax(result,axis=1)
        return result


class HiddenLayer:
    def __init__(self, x, num):
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1, 1, (columns, num))
        self.b = np.zeros([row, num], dtype=float)
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)
            for j in range(row):
                self.b[j, i] = rand_b
        h = self.sigmoid(np.dot(x, self.w) + self.b)
        self.H_ = np.linalg.pinv(h)
        # print(self.H_.shape)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def regressor_train(self, T):
        T = T.reshape(-1, 1)
        self.beta = np.dot(self.H_, T)
        return self.beta

    def classifisor_train(self, T):
        en_one = OneHotEncoder()
        T = en_one.fit_transform(T.reshape(-1, 1)).toarray()  # 独热编码之后一定要用toarray()转换成正常的数组
        # T = np.asarray(T)
        print(self.H_.shape)
        print(T.shape)
        self.beta = np.dot(self.H_, T)
        print(self.beta.shape)
        return self.beta

    def regressor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        return result

    def classifisor_test(self, test_x):
        b_row = test_x.shape[0]
        h = self.sigmoid(np.dot(test_x, self.w) + self.b[:b_row, :])
        result = np.dot(h, self.beta)
        result = [item.tolist().index(max(item.tolist())) for item in result]
        return result



if __name__ == '__main__':
    print(1)
    # a = RELM_HiddenLayer(X_train, j)
    # a.classifisor_train(Y_train)
    # predict = a.classifisor_test(X_test)