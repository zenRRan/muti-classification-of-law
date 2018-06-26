#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: prison_params.py
@time: 2018/5/27 17:20
"""

class Prison_params:

    def __init__(self):
        self.embed_num = 419996
        self.embed_dim = 100
        self.class_num = 9
        self.paddingId = 1
        self.pretrained_embed = False
        self.pretrained_weight = None
        self.embed_finetune = False
        self.dropout_emb = 0.2
        self.dropout = 0.4
        self.lstm_hiddens = 100
        self.lstm_layers = 1
        self.use_cuda = True