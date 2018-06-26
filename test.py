#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/5/15 12:48
"""

import numpy as np
import random
import torch
import time
import torch.nn as nn
from torch.autograd import Variable
import re


x = torch.FloatTensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]],
                       [[2, 2], [2, 2]], [[2, 2], [2, 2]], [[2, 2], [2, 2]],
                       [[3, 3], [3, 3]], [[3, 3], [3, 3]]])

print(x)
B = 3
D = x.size(2)
max_word_size = x.size(1)
sentence_count = [4, 3, 2]
max_sentence = int(np.max(sentence_count))
count = 0
j = 0
x = x.view(x.size(0), x.size(1) * x.size(2))
pad_final = torch.zeros(B * max_sentence, max_word_size * D).type(torch.FloatTensor)
# pad_final = torch.zeros(len(sentence_count), max_sentence * max_word_size, D).type(torch.FloatTensor)
print(pad_final)
for b in range(B):
    b_length = sentence_count[b]
    _sum = j + b_length
    pad_final[(max_sentence * b):(max_sentence * b + b_length)] = x[j:_sum]
    j = _sum
# print(pad_final)
pad_final = pad_final.view(B, max_sentence * max_word_size, D)
pad_final = Variable(pad_final)
# print(pad_final.data.tolist())
