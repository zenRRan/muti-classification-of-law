#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: test.py
@time: 2018/6/29 9:07
"""


from collections import Counter
import matplotlib.pyplot as plt
import numpy as  np
from math import log
import random

X = [[1, 1], [2, 2], [3, 3], [1, 3], [4, 4], [4, 5],
     [5, 3], [6, 4], [5, 5], [5, 4], [6, 7], [5, 6], [4, 6]]
Y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
X = np.array(X)
Y = np.array(Y)
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.ion()
ax = plt.figure().add_subplot(111)
ax.scatter(X[:4][:,0], X[:4][:,1], c='red')
ax.scatter(X[4:9][:,0], X[4:9][:,1], c='green')
ax.scatter(X[9:13][:,0], X[9:13][:,1], c='blue')
last_line = None

def split(X, Y, dim, value):
    index_a = (X[:, dim] <= value)
    index_b = (X[:, dim] > value)
    return X[index_a], X[index_b], Y[index_a], Y[index_b]

def entropy(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res += -p * log(p)
    return res

def gini(y):
    counter = Counter(y)
    res = 0.0
    for num in counter.values():
        p = num / len(y)
        res -= p**2
    return res

def try_split(X, Y, func, last_line, color):
    best_func = float('inf')
    best_dim, best_value = -1, -1
    for dim in range(X.shape[1]):
        sorted_index = np.argsort(X[:, dim])
        for i in range(1, len(X)):
            if X[sorted_index[i-1], dim] != X[sorted_index[i], dim]:
                value = (X[sorted_index[i-1], dim] + X[sorted_index[i], dim]) / 2
                X_l, X_r, Y_l, Y_r = split(X, Y, dim, value)
                e = func(Y_l) + func(Y_r)
                if e < best_func:
                    best_dim, best_value, best_func = dim, value, e
                    if dim == 0:
                        if last_line is not None:
                            ax.lines.pop(0)
                        last_line = ax.plot([best_value, best_value], [0, 8], color)
                        plt.pause(1)
                    else:
                        if len(ax.lines) > 1:
                            ax.lines.pop(1)
                        last_line = ax.plot([0, 8], [best_value, best_value], color)
                        plt.pause(1)
    return best_func, best_dim, best_value
func = entropy
colors = ['pink', 'purple']
for c in colors:
    best_func, dim, best_value = try_split(X, Y, func=func, last_line=last_line, color=c)
    X_l, X_r, Y_l, Y_r = split(X, Y, dim, best_value)
    print('best_value:', best_value)
    print(func(Y_l))
    print(func(Y_r))
    X = X_r
    Y = Y_r
# best_func, dim, best_value = try_split(X_r, Y_r, func=func, last_line=last_line)
# X_rl, X_rr, Y_rl, Y_rr = split(X_r, Y_r, dim, best_value)
# print('best_value:', best_value)
# print(func(Y_rl))
# print(func(Y_rr))
plt.pause(100)
