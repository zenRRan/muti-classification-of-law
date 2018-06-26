#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: train.py
@time: 2018/5/16 16:48
"""


from models.LSTM import Model as LSTM
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Common import *
from cail_eval import *
import time
torch.manual_seed(23)

def label2hot(label_list, label_alpha):
    batch_size = len(label_list)
    new_label = []
    for batch in range(batch_size):
        l = []
        l.extend([0 for _ in range(label_alpha.m_size)])
        new_label.append(l)
    for batch_idx in range(batch_size):
        for label in label_list[batch_idx]:
            new_label[batch_idx][label] = 1
    return new_label

def train(config, alpha_tuple, corpus_tuple):
    word_alpha, label_alpha = alpha_tuple
    train_data, train_label, dev_data, dev_label, test_data, test_label = corpus_tuple
    model = None
    if config.LSTM:
        model = LSTM(word_alpha, label_alpha, config)
    if model == None:
        print('model is None')
        raise RuntimeError
    if config.use_cuda:
        model = model.cuda()

    optimizer = None
    w_decay = 0
    if config.weight_decay:
        w_decay = config.decay
    if config.Adam:
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=w_decay)
    if config.SGD:
        optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=w_decay)
    if optimizer == None:
        print('optimizer is None')
        raise RuntimeError

    def test_accuracy(text_tuple, model):
        total_loss = torch.Tensor([0])
        if config.use_cuda:
            total_loss = total_loss.cuda()
        cnt = 0
        test_correct = 0
        for batch in create_batch_by_sorted(text_tuple, config.batch_size):
            text, label = unzip_batch(batch)
            text = seq2id(text, word_alpha, padding=True)
            label = seq2id(label, label_alpha)
            label = label2hot(label, label_alpha)
            text = Variable(torch.LongTensor(text))
            label = Variable(torch.LongTensor(label))
            if config.use_cuda:
                text = text.cuda()
                label = label.cuda()
            Y = model(text)
            Y = Y.view(Y.size(0) * Y.size(1), -1)
            label = label.view(label.size(0) * label.size(1), -1).squeeze(1)
            cnt += 1
            # if cnt % 500 == 0:
            #     print(cnt)
            test_correct += (torch.max(Y, 1)[1].view(label.size()).data == label.data).sum()
        test_acc = float(test_correct) / (len(text_tuple) * (label_alpha.m_size)) * 100
        return test_acc

    text_tuple = list(zip(train_data, train_label))
    dev_tuple = list(zip(dev_data, dev_label))
    test_tuple = list(zip(test_data, test_label))
    # print(model)
    # f_micro = Eval()
    # print(len(text_tuple[0][0]))
    # print(len(text_tuple[1][0]))
    for step in range(config.steps):
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        total_loss = torch.Tensor([0])
        if config.use_cuda:
            total_loss = total_loss.cuda()
        cnt = 0
        train_correct = 0
        start_time = time.time()
        for batch in create_batch_by_sorted(text_tuple, config.batch_size):
            text, label = unzip_batch(batch)
            # print(text)
            # print(label)
            text = seq2id(text, word_alpha, padding=True)
            label = seq2id(label, label_alpha)
            label = label2hot(label, label_alpha)
            text = Variable(torch.LongTensor(text))
            label = Variable(torch.LongTensor(label))
            if config.use_cuda:
                text = text.cuda()
                label = label.cuda()
            Y = model(text)
            # print(Y)
            label = label.view(label.size(0) * label.size(1), -1).squeeze(1)
            # Micro_F1_measure(Y, label, f_micro)
            Y = Y.view(Y.size(0)*Y.size(1), -1)
            loss = F.cross_entropy(Y, label)
            loss.backward()
            optimizer.step()
            cnt += 1
            if cnt % 1 == 0:
                end_time = time.time() - start_time
                mins, secs = sec2min(end_time)
                print('step{}/{} {:.2f}% {:.1f}s'.format(step, config.steps, cnt/(len(text_tuple)//config.batch_size+1)*100, end_time))
            total_loss += loss.data
            # print('Y max idx:', torch.max(Y, 1)[1].view(label.size()).data)
            # print('label:', label.data)
            train_correct += (torch.max(Y, 1)[1].view(label.size()).data == label.data).sum()
            # print(train_correct)
        if config.lr_decay:
            adjust_learning_rate(optimizer, config.lr / (1 + step * config.decay))
        total_loss /= len(train_data)
        total_loss = total_loss.cpu()
        # loss_list.append(total_loss.numpy()[0])
        # print(train_correct)
        # print(len(train_data))
        # print(label_alpha.m_size)
        train_acc = float(train_correct) / (len(train_data) * (label_alpha.m_size)) * 100
        dev_acc = test_accuracy(dev_tuple, model)
        test_acc = test_accuracy(test_tuple, model)
        print('step:', step, 'loss:', total_loss.numpy()[0], 'train:{:.6f}%'.format(train_acc))
        # print('step:', step, 'loss:', total_loss.numpy()[0], 'train:', train_acc, ' #dev:', dev_acc, ' #test:', test_acc)




def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sec2min(secends):
    mins = secends // 60
    sec = secends % 60
    return mins, sec














