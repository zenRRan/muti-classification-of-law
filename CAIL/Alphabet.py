#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Alphabet.py
@time: 2018/5/16 14:44
"""

import collections

class Alphabet:
    def __init__(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = collections.OrderedDict()

    def initial(self, stat, cutoff=0):
        for key in stat:
            if stat[key] > cutoff:
                self.from_string(key)
        self.m_b_fixed = True

    def from_string(self, str):
        if str in self.string2id:
            return self.string2id[str]
        else:
            if not self.m_b_fixed:
                newid = self.m_size
                self.string2id[str] = newid
                self.id2string.append(str)
                self.m_size += 1
                if self.m_size >= self.max_cap:
                    self.m_b_fixed = True
                return newid
            else:
                return -1

    def from_id(self, id, definedStr=''):
        if id >= self.m_size or int(id) < 0:
            return definedStr
        else:
            return self.id2string[id]
    def write(self, path):
        fopen = open(path, encoding="utf-8", mode='w')
        for key in self.string2id:
            fopen.write(key+" "+str(self.string2id[key])+'\n')
        fopen.close()

    def read(self, path):
        fopen = open(path, encoding="utf-8", mode='r')
        for line in fopen:
            info = line.strip().split(" ")
            if not info[1].isdigit():
                print(line[1], ' not !digit!')
                raise RuntimeError
            self.string2id[info[0]] = int(info[1])
            self.id2string.append(info[0])
        self.m_b_fixed = True
        self.m_size = len(self.string2id)
        fopen.close()

    def clean(self):
        self.max_cap = 1e8
        self.m_size = 0
        self.m_b_fixed = False
        self.id2string = []
        self.string2id = collections.OrderedDict()

    def set_fixed_flag(self, bfixed):
        self.m_b_fixed = bfixed
        if (not self.m_b_fixed) and (self.m_size >= self.max_cap):
            self.m_b_fixed = True
