#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: main.py
@time: 2018/5/15 9:04
"""

from read import Reader_Json, read_file
from config import configer
from Alphabet import Alphabet
from train import train
import collections
from Common import *
import argparse


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', default='./config.cfg', help='input the config path')
    parse.add_argument('--use_cuda', default=False, help='if use cuda or not , default is false')
    args = parse.parse_args()
    config = configer(args.config)
    train_data = read_file(config.train_data_path, split=True)
    train_label = read_file(config.train_label_path, split=True)
    test_data = read_file(config.test_data_path, split=True)
    test_label = read_file(config.test_label_path, split=True)
    dev_data = read_file(config.dev_data_path, split=True)
    dev_label = read_file(config.dev_label_path, split=True)

    word_alpha, label_alpha = createAlphabet(train_data, train_label, config.word_cut_off)
    alpha_tuple = (word_alpha, label_alpha)
    corpus_tuple = (train_data, train_label, dev_data, dev_label, test_data, test_label)
    train(config, alpha_tuple, corpus_tuple)







