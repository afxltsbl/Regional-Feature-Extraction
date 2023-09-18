import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, auc
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix, classification_report


class Metric(object):
    def __init__(self, output, label):
        self.output = output  # prediction label matric
        self.label = label  # true  label matric

    def accuracy_all(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def accuracy_mean(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        accuracy = np.mean(np.equal(y_true, y_pred))  # 需要是numpy数组
        return accuracy

    def micfscore(self, thresh=0.5, type='micro'):  # micro_f1
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def macfscore(self, thresh=0.5, type='macro'):  # macro_f1
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def hanming_distanc(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return hamming_loss(y_true, y_pred)

    def fscore(self, type='micro'):
        y_pred = self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred, 1), np.argmax(y_true, 1), average=type)

    def auc(self):
        try:
            aucc = roc_auc_score(self.label, self.output, average=None)
            return aucc
        except ValueError:
            pass

    def recall(self, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        TP = np.sum(y_true * y_pred)
        FN = np.sum(y_true * (1 - y_pred))
        recall = TP / (TP + FN)
        return recall

    def classification_report(self, labels_name, thresh=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return classification_report(y_true, y_pred, target_names=labels_name)

    def multilabel_confusion_matrix(self, thresh=0.5):
        """计算多标签混淆矩阵"""
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > thresh, 1, 0)
        return multilabel_confusion_matrix(y_true, y_pred)

    def auROC(self):
        y_pred = self.output
        y_true = self.label
        row, col = y_true.shape
        temp = []
        ROC = 0
        for i in range(col):
            sigle_ROC = roc_auc_score(y_true[:, i], y_pred[:, i], average='macro', sample_weight=None)
            # print("%d th AUROC: %f"%(i,ROC))
            temp.append(sigle_ROC)
            ROC += sigle_ROC
        return ROC / (col)

    def MacroAUC(self):
        y_pred = self.output  # num_instance*num_label
        y_true = self.label  # num_instance*num_label
        num_instance, num_class = y_pred.shape
        count = np.zeros((num_class, 1))  # store the number of postive instance'score>negative instance'score
        num_P_instance = np.zeros((num_class, 1))  # number of positive instance for every label
        num_N_instance = np.zeros((num_class, 1))
        AUC = np.zeros((num_class, 1))  # for each label
        count_valid_label = 0
        for i in range(num_class):  # 第i类
            num_P_instance[i, 0] = sum(y_true[:, i] == 1)  # label,,test_target
            num_N_instance[i, 0] = num_instance - num_P_instance[i, 0]
            # exclude the label on which all instances are positive or negative,
            # leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
            if num_P_instance[i, 0] == 0 or num_N_instance[i, 0] == 0:
                AUC[i, 0] = 0
                count_valid_label = count_valid_label + 1
            else:

                temp_P_Outputs = np.zeros((int(num_P_instance[i, 0]), num_class))
                temp_N_Outputs = np.zeros((int(num_N_instance[i, 0]), num_class))
                #
                temp_P_Outputs[:, i] = y_pred[y_true[:, i] == 1, i]
                temp_N_Outputs[:, i] = y_pred[y_true[:, i] == 0, i]
                for m in range(int(num_P_instance[i, 0])):
                    for n in range(int(num_N_instance[i, 0])):
                        if (temp_P_Outputs[m, i] > temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 1
                        elif (temp_P_Outputs[m, i] == temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 0.5

                AUC[i, 0] = count[i, 0] / (num_P_instance[i, 0] * num_N_instance[i, 0])
        macroAUC1 = sum(AUC) / (num_class - count_valid_label)
        return float(macroAUC1), AUC

    def avgPrecision(self):
        y_pred = self.output
        y_true = self.label
        num_instance, num_class = y_pred.shape
        precision_value = 0
        precisions = []
        for i in range(num_instance):
            p = precision_score(y_true[i, :], y_pred[i, :])
            precisions.append(p)
            precision_value += p
            # print(precision_value)
        pre_list = np.array([1.0] + precisions + [0.0])  # for get AUPRC
        # print(pre_list)
        return float(precision_value / num_instance), pre_list

    def avgRecall(self):
        y_pred = self.output
        y_true = self.label
        num_instance, num_class = y_pred.shape
        recall_value = 0
        recalls = []
        for i in range(num_instance):
            p = recall_score(y_true[i, :], y_pred[i, :])
            recalls.append(p)
            recall_value += p
        rec_list = np.array([0.0] + recalls + [1.0])  # for get AUPRC
        sorting_indices = np.argsort(rec_list)
        # print(rec_list)
        return float(recall_value / num_instance), rec_list, sorting_indices

    def getAUPRC(self):
        avgPrecision, precisions = self.avgPrecision()
        avfRecall, recalls, sorting_indices = self.avgRecall()
        # x is either increasing or decreasing
        # such as recalls[sorting_indices]
        auprc = auc(recalls[sorting_indices], precisions[sorting_indices])
        return auprc

    def cal_single_label_micro_auc(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        m = 0
        n = 0
        auc = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
                auc += n
            if y[i] == 0:
                n += 1
        auc /= (m * n)
        return auc

    def get_micro_auc(self):
        x = self.output
        y = self.label
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        x = x.reshape(n * d)
        y = y.reshape(n * d)
        auc = self.cal_single_label_micro_auc(x, y)
        return auc

    def cal_single_instance_coverage(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        loc = x.shape[0]
        for i in range(x.shape[0]):
            if y[i] == 1:
                loc -= i
                break
        return loc

    def get_coverage(self):
        x = self.output
        y = self.label
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        cover = 0
        for i in range(n):
            cover += self.cal_single_instance_coverage(x[i], y[i])
        cover = cover / n - 1
        return cover

    def cal_single_instance_ranking_loss(self, x, y):
        idx = np.argsort(x)  # 升序排列
        y = y[idx]
        m = 0
        n = 0
        rl = 0
        for i in range(x.shape[0]):
            if y[i] == 1:
                m += 1
            if y[i] == 0:
                rl += m
                n += 1
        rl /= (m * n)
        return rl

    def get_ranking_loss(self):
        x = self.output
        y = self.label
        n, d = x.shape
        if x.shape[0] != y.shape[0]:
            print("num of  instances for output and ground truth is different!!")
        if x.shape[1] != y.shape[1]:
            print("dim of  output and ground truth is different!!")
        m = 0
        rank_loss = 0
        for i in range(n):
            s = np.sum(y[i])
            if s in range(1, d):
                rank_loss += self.cal_single_instance_ranking_loss(x[i], y[i])
                m += 1
        rank_loss /= m
        return rank_loss

    def train_accuracy(y_hat, y):
        return (y_hat.argmax(dim=1) == y).float().mean().item()
