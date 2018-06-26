#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: read.py
@time: 2018/5/15 9:03
"""

import json
import jieba
import random
from collections import Counter

class Reader_Json:
    def __init__(self, path):
        self.data_lists = []
        self.accusations = Counter()
        self.relevant_articles = Counter()
        self.imprisonments = Counter()
        self.sent_len = Counter()
        with open(path, 'r', encoding='utf8') as f:
            accusation_set = {}
            relevant_article_set = {}
            imprisonment_set = {}
            for line in f.readlines():
                dic = json.loads(line)
                new_data = []
                fact = dic['fact']
                meta_dic = dic['meta']
                relevant_articles = '#'.join([str(elem) for elem in meta_dic['relevant_articles']])
                accusation = meta_dic['accusation']
                term_dic = meta_dic['term_of_imprisonment']
                death_penalty = term_dic['death_penalty']
                imprisonment = term_dic['imprisonment']
                life_imprisonment = term_dic['life_imprisonment']
                imprisonment = [str(death_penalty), str(imprisonment), str(life_imprisonment)]

                new_data.append(fact.lower())
                self.sent_len[len(fact)] += 1
                new_data.append(relevant_articles)
                # new_data.append(accusation)
                # new_data.append(meta_dic['punish_of_money'])
                # new_data.append(meta_dic['criminals'])
                # new_data.append(imprisonment)

                self.relevant_articles[relevant_articles] += 1
                # self.accusations[accusation] += 1
                # self.imprisonments[imprisonment] += 1

                self.data_lists.append(new_data)

        # print(self.relevant_articles)
        # print(self.accusations)
        # print(self.imprisonments)
        # print(self.data_lists[:10])
        cnt_big1k = 0
        cnt_sml1k = 0
        for elem in self.sent_len:
            if elem > 2999:
                cnt_big1k += self.sent_len[elem]
            else:
                cnt_sml1k += self.sent_len[elem]
        print(cnt_big1k)
        print(cnt_sml1k)
        print(self.sent_len)
    def get_data_lists(self):
        return self.data_lists

def read_file(path, split=False):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        lines = [random.choice(lines) for _ in range(50)]
        for line in lines:
            line = line.strip()
            if split:
                line = line.split()[:50]
                data.append(line)
            else:
                data.append([line])
    return data


class reader:

    def __init__(self, readpath):
        self.data = []
        with open(readpath, 'r', encoding="utf-8") as f:
            for i in f.readlines():
                temp = json.loads(i)
                list = jieba.cut(temp['fact'].lower(), cut_all=False)
                fact = " ".join(list)
                data = temp['meta']['accusation']
                accusation = data[0]
                self.data.append(accusation+"|||"+fact)
            f.close()

    def getdata(self):
            return self.data

    def writefile(self, path):
        with open(path, 'a', encoding="utf-8") as w:
            for i in self.getdata():
                w.write(i + '\n')
            w.close()








train_path = './corpus/cail2018_small/good/data_train.json'
read = Reader_Json(train_path)




# valid_path = './corpus/cail2018_small/good/data_valid.json'
# test_path = './corpus/cail2018_small/good/data_test.json'

# raw_train_facts_path = './corpus/cail2018_small/good/fact_raw_train.txt'
# raw_valid_facts_path = './corpus/cail2018_small/good/fact_raw_valid.txt'
# raw_test_facts_path = './corpus/cail2018_small/good/fact_raw_test.txt'
#
# train_acst_path = './corpus/cail2018_small/good/acst_train.txt'
# valid_acst_path = './corpus/cail2018_small/good/acst_valid.txt'
# test_acst_path = './corpus/cail2018_small/good/acst_test.txt'
#
# train_seg_facts_path = './corpus/cail2018_small/good/fact_seg_train.txt'
# valid_seg_facts_path = './corpus/cail2018_small/good/fact_seg_valid.txt'
# test_seg_facts_path = './corpus/cail2018_small/good/fact_seg_test.txt'
# train_read = Reader(train_path)
# valid_read = Reader(valid_path)
# test_read = Reader(test_path)

# write_raw2file = False
# if write_raw2file:
#     print('write raw to file...')
#     with open(raw_train_facts_path, 'w', encoding='utf8') as f1, open(train_acst_path, 'w', encoding='utf8') as f2:
#         for elem in train_read.get_data_lists():
#             f1.write(elem[0] + '\n')
#             f2.write(' '.join(elem[2]) + '\n')
#
#     with open(raw_valid_facts_path, 'w', encoding='utf8') as f1, open(valid_acst_path, 'w', encoding='utf8') as f2:
#         for elem in valid_read.get_data_lists():
#             f1.write(elem[0] + '\n')
#             f2.write(' '.join(elem[2]) + '\n')
#
#     with open(raw_test_facts_path, 'w', encoding='utf8') as f1, open(test_acst_path, 'w', encoding='utf8') as f2:
#         for elem in test_read.get_data_lists():
#             f1.write(elem[0] + '\n')
#             f2.write(' '.join(elem[2]) + '\n')
#
# segment = False
# if segment:
#     print('segment the sentences...')
#     with open(raw_train_facts_path, 'r', encoding='utf8') as f1, open(train_seg_facts_path, 'w', encoding='utf8') as f2:
#         for line in f1.readlines():
#             f2.write(' '.join(jieba.cut(line)))
#
#     with open(raw_valid_facts_path, 'r', encoding='utf8') as f1, open(valid_seg_facts_path, 'w', encoding='utf8') as f2:
#         for line in f1.readlines():
#             f2.write(' '.join(jieba.cut(line)))
#
#     with open(raw_test_facts_path, 'r', encoding='utf8') as f1, open(test_seg_facts_path, 'w', encoding='utf8') as f2:
#         for line in f1.readlines():
#             f2.write(' '.join(jieba.cut(line)))

