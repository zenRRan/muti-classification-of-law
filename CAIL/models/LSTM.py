#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: LSTM.py
@time: 2018/5/15 9:05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(23)

class Model(nn.Module):
    def __init__(self, vacab_alpha, label_alpha, config):
        super(Model, self).__init__()
        self.vacab_size = vacab_alpha.m_size
        self.label_size = label_alpha.m_size
        self.embed_dim = config.embed_size
        self.hidden_size = config.hidden_size
        self.hidden_layers = config.hidden_layers
        self.use_cuda = config.use_cuda
        self.bidirection = config.bidirection
        self.batch_size = config.batch_size
        self.dropout = config.dropout
        if self.bidirection:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.dropout = nn.Dropout(config.dropout)
        self.embedding = nn.Embedding(self.vacab_size, self.embed_dim)
        self.lstm = nn.LSTM(input_size=self.embed_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.hidden_layers,
                            bidirectional=self.bidirection)
        self.weight = nn.Parameter(torch.randn(self.label_size, self.hidden_size*2))
        self.linear1 = nn.Linear(self.hidden_size*2, self.label_size)
        self.linear2 = nn.Linear(self.hidden_size*2, 2)

    def forward(self, input):
        batch_size = input.size(0)
        sent_size = input.size(1)

        input = self.embedding(input) # (batch, sent_size, embed_size)
        # print('embedding:', input.size())
        # input = self.dropout(input) # (batch, sent_size, embed_size)
        input, _ = self.lstm(input) # (sent_size, batch, hidden_size*bidirection)
        # print('lstm:', input.size())
        input = torch.transpose(input, 1, 2)
        # print('tranpose:', input.size())
        input = torch.tanh(input)
        input = F.max_pool1d(input, input.size(2))
        # print('pooling:', input.size())
        # print('hidden:', input)
        input = input.view(batch_size, 1, self.hidden_size*self.num_direction).\
                expand(batch_size, self.label_size, self.hidden_size*self.num_direction)
        # print(input.size())
        # print('expand:', input.size())
        # input = self.linear1(input)
        input = input * self.weight
        # print('linear1:', input)
        input = torch.tanh(input)
        input = self.linear2(input)
        # print('linear2:', input)
        return input



















