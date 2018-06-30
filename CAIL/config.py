#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: config.py
@time: 2018/5/15 9:04
"""


from configparser import ConfigParser

class configer:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.read(path)

    @property
    def train_data_path(self):
        return self.config.get('path', 'train_data_path')

    @property
    def train_label_path(self):
        return self.config.get('path', 'train_label_path')

    @property
    def dev_data_path(self):
        return self.config.get('path', 'dev_data_path')

    @property
    def dev_label_path(self):
        return self.config.get('path', 'dev_label_path')

    @property
    def test_data_path(self):
        return self.config.get('path', 'test_data_path')

    @property
    def test_label_path(self):
        return self.config.get('path', 'test_label_path')

    @property
    def lr(self):
        return self.config.getfloat('data', 'lr')

    @property
    def steps(self):
        return self.config.getint('data', 'steps')

    @property
    def Adam(self):
        return self.config.getboolean('data', 'Adam')

    @property
    def SGD(self):
        return self.config.getboolean('data', 'SGD')

    @property
    def dropout(self):
        return self.config.getfloat('data', 'dropout')

    @property
    def embed_size(self):
        return self.config.getint('data', 'embed_size')

    @property
    def hidden_size(self):
        return self.config.getint('data', 'hidden_size')

    @property
    def hidden_layers(self):
        return self.config.getint('data', 'hidden_layers')

    @property
    def bidirection(self):
        return self.config.getboolean('data', 'bidirection')

    @property
    def batch_size(self):
        return self.config.getint('data', 'batch_size')

    @property
    def word_cut_off(self):
        return self.config.getint('data', 'word_cut_off')

    @property
    def weight_decay(self):
        return self.config.getboolean('data', 'weight_decay')

    @property
    def decay(self):
        return self.config.getfloat('data', 'decay')

    @property
    def lr_decay(self):
        return self.config.getboolean('data', 'lr_decay')

    @property
    def LSTM(self):
        return self.config.getboolean('data', 'LSTM')

    @property
    def use_cuda(self):
        return self.config.getboolean('data', 'use_cuda')