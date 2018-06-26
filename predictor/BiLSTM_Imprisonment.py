# @Author : bamtercelboo
# @Datetime : 2018/1/31 9:24
# @File : model_PNC.py
# @Last Modify Time : 2018/1/31 9:24
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  model_PNC.py
    FUNCTION : Part-of-Speech Tagging(POS), Named Entity Recognition(NER) and Chunking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable as Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import torch.nn.init as init
import numpy as np
import time

from wheel.signatures import assertTrue

torch.manual_seed(233)
random.seed(233)


class BiLSTM_Imprisonment(nn.Module):

    def __init__(self, config):
        super(BiLSTM_Imprisonment, self).__init__()
        self.config = config
        self.cat_size = 5

        V = config.embed_num
        D = config.embed_dim
        C = config.class_num
        paddingId = config.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        # self.embed = nn.Embedding(V, D)

        if config.pretrained_embed:
            self.embed.weight.data.copy_(config.pretrained_weight)
        self.embed.weight.requires_grad = self.config.embed_finetune

        self.dropout_embed = nn.Dropout(config.dropout_emb)
        self.dropout = nn.Dropout(config.dropout)

        # self.weight = nn.Parameter(torch.randn(C, config.lstm_hiddens), requires_grad=True)
        # self.weight = nn.Parameter(torch.randn(C, config.lstm_hiddens * 2), requires_grad=True)
        # init.xavier_uniform(self.weight)
        self.bilstm = nn.LSTM(input_size=D, hidden_size=config.lstm_hiddens, num_layers=config.lstm_layers,
                              bidirectional=True, bias=True)
        # self.init_lstm()
        # if self.config.use_cuda is True:
        #     self.bigru.cuda()

        self.linear = nn.Linear(in_features=config.lstm_hiddens * 2, out_features=C, bias=True)
        # self.linear = nn.Linear(in_features=config.lstm_hiddens, out_features=2, bias=True)
        init.xavier_uniform(self.linear.weight)
        bias_value = np.sqrt(6 / (config.lstm_hiddens + 1))
        self.linear.bias.data.uniform_(-bias_value, bias_value)

    def context_embed(self, embed, batch_features):
        context_indices = batch_features.context_indices
        B, T, WS = context_indices.size()
        B_embed, T_embed, dim = embed.size()
        if assertTrue((B == B_embed) and (T == T_embed)) is False:
            print("invalid argument")
            exit()
        context_indices = context_indices.view(B, T * WS)
        if self.config.use_cuda is True:
            context_np = context_indices.data.cpu().numpy()
        else:
            context_np = np.copy(context_indices.data.numpy())
        for i in range(B):
            for j in range(T * WS):
                context_np[i][j] = T * i + context_np[i][j]
        if self.config.use_cuda is True:
            context_indices = Variable(torch.from_numpy(context_np)).cuda()
        else:
            context_indices = Variable(torch.from_numpy(context_np))
        context_indices = context_indices.view(context_indices.size(0) * context_indices.size(1))

        embed = embed.view(B * T, dim)
        if self.config.use_cuda is True:
            pad_embed = Variable(torch.zeros(1, dim)).cuda()
        else:
            pad_embed = Variable(torch.zeros(1, dim))
        embed = torch.cat((embed, pad_embed), 0)
        context_embed = torch.index_select(embed, 0, context_indices)
        context_embed = context_embed.view(B, T, -1)

        return context_embed

    def forward(self, sorted_inputs_words, sorted_seq_lengths, desorted_indices):

        word = sorted_inputs_words
        sentence_length = sorted_seq_lengths
        # print(sentence_length)
        x = self.embed(word)  # (N,W,D)
        # context_embed = self.context_embed(x, batch_features)
        x = self.dropout_embed(x)
        # context_embed = self.dropout_embed(context_embed)
        # print(context_embed)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        # print(packed_embed)
        # print(sentence_length)
        x, _ = self.bilstm(packed_embed)
        # x, _ = self.bigru(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[desorted_indices]
        # print(x.permute(0, 2, 1).size())
        x = x.permute(0, 2, 1)
        # x = F.tanh(x)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        x = F.tanh(x)
        logit = self.linear(x)
        return logit

