# @Author : bamtercelboo
# @Datetime : 2018/5/17 8:39
# @File : cail_eval.py
# @Last Modify Time : 2018/5/17 8:39
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  cail_eval.py
    FUNCTION : None
"""

import os
import sys
import torch
from torch.autograd import Variable
import numpy as np

from Common import *


class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear_PRF(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = (2 * (self.precision * self.recall)) / (self.precision + self.recall)

        self.precision = np.round(self.precision, 4)
        self.recall = np.round(self.recall, 4)
        self.fscore = np.round(self.fscore, 4)

        return self.precision, self.recall, self.fscore

    def acc(self):
        return np.round((self.correct_num / self.predict_num) * 100, 4)


def Micro_F1_measure(logit, gold, f_micro):
    print("F-Score Micro")
    # f_micro = Eval()

    # fixed the random state
    np.random.seed(233)
    torch.manual_seed(233)

    """
        batch_size = 3
        class_num = 8
        label_size = 2
    """
    # logit = Variable(torch.randn(3, 8, 2).type(torch.FloatTensor))
    # gold = Variable(torch.from_numpy(np.random.randint(low=0, high=2, size=3 * 8)).type(torch.LongTensor))
    # gold = torch.(3 * 8)
    # print("logit", logit)
    # print("gold", gold)

    predict_batch_list = getMaxindex_batch(logit.view(logit.size(0) * logit.size(1), -1))
    gold_list = gold.data.tolist()
    # print(len(maxId_batch))
    # print("maxId_batch", maxId_batch)
    # print(maxId_batch)

    maxId = Variable(torch.from_numpy(np.array(predict_batch_list)).type(torch.LongTensor))
    # print(maxId)

    print("predict", predict_batch_list)
    print("gold   ", gold.data.tolist())
    correct_num = 0
    assert len(predict_batch_list) == len(gold_list)
    for p, g in zip(predict_batch_list, gold_list):
        if (p == g) and (p != 0):
            correct_num += 1
    # correct_num = (maxId.data == gold.data).sum()
    predict_num = predict_batch_list.count(1)
    gold_num = gold_list.count(1)

    f_micro.correct_num = correct_num
    f_micro.predict_num = predict_num
    f_micro.gold_num = gold_num

    print(correct_num, predict_num, gold_num)
    p, r, f = f_micro.getFscore()
    print(p, r, f)
    # print(f_micro.acc())


def MAcro_F1_measure():
    print("F-Score Macro")
    f_macro = Eval()

    # fixed the random state
    np.random.seed(233)
    torch.manual_seed(233)

    """
        batch_size = 3
        class_num = 8
        label_size = 2
    """
    logit = Variable(torch.randn(3, 8, 2).type(torch.FloatTensor))
    gold = Variable(torch.from_numpy(np.random.randint(low=0, high=2, size=3 * 8)).type(torch.LongTensor))





# if __name__ == "__main__":
#     print("CAIL2018 F-Score(micro, macro)")
#     Micro_F1_measure()




