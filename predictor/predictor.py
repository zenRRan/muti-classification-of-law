import torch
import jieba
from torch.autograd import Variable
from BiLSTM import BiLSTM
from BiLSTM_Imprisonment import BiLSTM_Imprisonment
from accu_params import Accu_params
from law_params import Law_params
from prison_params import Prison_params

class Predictor:

    def create_word_dic(self, path):
        word_dic = {}
        word_dic_file = open(path, 'r', encoding='utf8')
        for line in word_dic_file.readlines():
            line = line.strip().split()
            word_dic[line[0]] = int(line[1])
        return word_dic

    def create_label_dic(self, path):
        label_dic = {}
        label_dic_file = open(path, 'r', encoding='utf8')
        for line in label_dic_file.readlines():
            line = line.strip().split()
            label_dic[int(line[1])] = line[0]
        return label_dic

    def load_model(self, NN, path, params):
        model = NN(params)
        # print(accu_model)
        model.load_state_dict(torch.load(path))
        # print(accu_model)
        return model

    def read_file(self, path):
        accu_file = open(path, 'r', encoding='utf8')
        idx = 1
        accu_dic = {}
        for line in accu_file.readlines():
            line = line.strip()
            accu_dic[line] = idx
            idx += 1
        return accu_dic

    def __init__(self):
        self.batch_size = 16
        #model path
        self.accu_model_path = './PredictModel_3k_0527/accu/cail_accu_model_epoch_132.pt'
        self.law_model_path = './PredictModel_3k_0527/law/cail_law_model_epoch_210.pt'
        self.prison_model_path = './PredictModel_3k_0527/prison/cail_model_epoch_5.pt'
        #word path
        self.word_dic_path = './PredictModel_3k_0527/accu/dictionary_word.txt'
        #label path
        self.accu_label_dic_path = './PredictModel_3k_0527/accu/dictionary_label.txt'
        self.law_label_dic_path = './PredictModel_3k_0527/law/dictionary_label.txt'
        self.prison_label_dic_path = './PredictModel_3k_0527/prison/dictionary_label.txt'
        self.accu_path = './accu.txt'
        self.law_path = './law.txt'
        self.word_dic = {}
        self.accu_label_dic = {}
        self.law_label_dic = {}
        self.prison_label_dic = {}

        self.PAD = '<pad>'
        self.unk = '<unk>'
        self.accu_dic = {}
        self.law_dic = {}
        self.use_cuda = False
        '''
            load accu, law's file to dic
        '''
        self.accu_dic = self.read_file(self.accu_path)
        self.law_dic = self.read_file(self.law_path)

        '''
            load model
        '''
        # print("load accu model")
        self.accu_model = self.load_model(BiLSTM, self.accu_model_path, Accu_params())
        # print('done')
        # print("load law model")
        self.law_model = self.load_model(BiLSTM, self.law_model_path, Law_params())
        # print('done')
        # print("load prison model")
        self.prison_model = self.load_model(BiLSTM_Imprisonment, self.prison_model_path, Prison_params())
        # print('done')
        if self.use_cuda:
            self.accu_model = self.accu_model.cuda()
            self.law_model = self.law_model.cuda()
            self.prison_model = self.prison_model.cuda()

        self.word_dic = self.create_word_dic(self.word_dic_path)
        self.accu_label_dic = self.create_label_dic(self.accu_label_dic_path)
        self.law_label_dic = self.create_label_dic(self.law_label_dic_path)
        self.prison_label_dic = self.create_label_dic(self.prison_label_dic_path)

    def prepare_pack_padded_sequence(self, inputs_words, seq_lengths, descending=True):
        sorted_seq_lengths, indices = torch.sort(torch.LongTensor(seq_lengths), descending=descending)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_inputs_words = inputs_words[indices]
        return sorted_inputs_words, sorted_seq_lengths.numpy(), desorted_indices

    def predict(self, content):
        result = []

        content_list = []
        for line in content:
            line = line.strip()
            line = jieba.cut(line)
            line = ' '.join(line)
            content_list.append(line)
        content_idx_list, sentence_length = self.seq2id(content_list)
        # print(content_idx_list)
        # print(sentence_length)
        #predict accu
        content_idx_list = Variable(torch.LongTensor(content_idx_list))

        sorted_inputs_words, sorted_seq_lengths, desorted_indices = self.prepare_pack_padded_sequence(content_idx_list,
                                                                                                      sentence_length)
        if self.use_cuda:
            sorted_inputs_words = sorted_inputs_words.cuda()
            desorted_indices = desorted_indices.cuda()
        accu_label_idxs = self.pred_accu(sorted_inputs_words, sorted_seq_lengths, desorted_indices)
        law_label_idxs = self.pred_law(sorted_inputs_words, sorted_seq_lengths, desorted_indices)
        prison_label_idxs = self.pred_prison(sorted_inputs_words, sorted_seq_lengths, desorted_indices)

        for i, _ in enumerate(content):
            result.append({
                    "accusation": accu_label_idxs[i],
                    "imprisonment": prison_label_idxs[i],
                    "articles": law_label_idxs[i]
                })

        # for a in range(0, len(content)):
        #     result.append({
        #         "accusation": [1, 2, 3],
        #         "imprisonment": 5,
        #         "articles": [5, 7, 9]
        #     })
        return result

    def pred_accu(self, sorted_inputs_words, sorted_seq_lengths, desorted_indices):

        pred = self.accu_model(sorted_inputs_words, sorted_seq_lengths, desorted_indices)
        label_idxs = []
        for l in torch.max(pred, 2)[1].data.tolist():
            new = []
            for i, x in enumerate(l):
                if x == 1:
                    new.append(i)
            label_idxs.append(new)
        accus_list = []
        for id_list in label_idxs:
            new_elem = []
            for idx in id_list:
                new_elem.append(self.accu_label_dic[idx])
            accus_list.append(new_elem)
        # print(accus_list)
        label_idxs = []
        for accus in accus_list:
            ids_list = []
            for accu in accus:
                ids_list.append(self.accu_dic[accu])
            label_idxs.append(ids_list)
        return label_idxs

    def pred_law(self, sorted_inputs_words, sorted_seq_lengths, desorted_indices):
        pred = self.law_model(sorted_inputs_words, sorted_seq_lengths, desorted_indices)
        label_idxs = []
        for l in torch.max(pred, 2)[1].data.tolist():
            new = []
            for i, x in enumerate(l):
                if x == 1:
                    new.append(i)
            label_idxs.append(new)
        law_list = []
        for id_list in label_idxs:
            new_elem = []
            for idx in id_list:
                new_elem.append(self.law_label_dic[idx])
            law_list.append(new_elem)
        # print(law_list)
        label_idxs = []
        for laws in law_list:
            ids_list = []
            for law in laws:
                ids_list.append(self.law_dic[law])
            label_idxs.append(ids_list)
        return label_idxs

    def pred_prison(self, sorted_inputs_words, sorted_seq_lengths, desorted_indices):
        pred = self.prison_model(sorted_inputs_words, sorted_seq_lengths, desorted_indices)
        label_idxs = torch.max(pred, 1)[1].data.tolist()
        prison_list = []
        for idx in label_idxs:
            prison_list.append(self.prison_label_dic[idx])
        # print(prison_list)
        label_idxs = []
        for prison in prison_list:
            label_idxs.append(self.every_months(prison))
        return label_idxs

    def every_months(self, y):
        if y == 0:
            return -2
        if y == 1:
            return -1
        if y == 2:
            return 120
        if y == 3:
            return 102
        if y == 4:
            return 72
        if y == 5:
            return 48
        if y == 6:
            return 30
        if y == 7:
            return 18
        else:
            return 6

    def seq2id(self, content):
        result = []
        sentence_length = []
        max_len = 0
        for line in content:
            # print(line)
            line = line.strip().split()
            sentence_length.append(len(line))
            max_len = max(max_len, len(line))
        for line in content:
            new_list = []
            line = line.strip().split()
            for word in line:
                if word in self.word_dic:
                    new_list.append(self.word_dic[word])
                else:
                    new_list.append(self.word_dic[self.unk])
            for _ in range(max_len-len(line)):
                # new_list.append(self.PAD)
                new_list.append(self.word_dic[self.PAD])
            result.append(new_list)
        return result, sentence_length

predictor = Predictor()

while(True):
    content = ['公诉机关指控：一、故意伤害2013年被告人陈某借给被害人张某某1万元，张某某归还1千元，仍有9千元没有归还。2014年11月12日10时许，\
               被告人陈某打电话给张某某要求其还钱，张某某认为陈某在逼其还钱，两人在电话中发生争吵，之后张某某打电话给陈某，要其去兴国县“钻石金店”，\
               将自己的戒指抵押掉还钱给陈某，陈某担心到“钻石金店”会和张某某打架，便打电话叫“小陈”、“小谢”、“小谢”（均另案处理）一起去“钻石金店”，\
               如果和张某某打架，他们也可以帮忙打架。到了“钻石金店”后，陈某和张某某再次发生争吵，张某某推了陈某肩膀一下，\
               陈某打了张某某胸部两三下，陈某让“小谢”打张某某，“小谢”跑到附近一个卖水果的摊位上拿起一把长约25厘米、宽约5厘米的水果刀，\
               朝张某某头部砍了三、四刀，张某某用手挡头部时手部也被砍伤。经鉴定，被害人张某某的损伤程度为轻伤二级。二、寻衅滋事2014年年初，\
               被告人陈某与被害人钟某的表弟肖某甲之间发生交通事故，钟某得知肇事者是陈某便给其打电话要求处理此事。\
               2014年11月22日晚22时许，被告人陈某和“小肖”（另案处理）在兴国县“星座网吧”上网时遇到被害人钟某也在上网，陈某要求钟某给他道歉，\
               钟某拒绝道歉，两人发生争吵，钟某打电话叫人，陈某也打电话叫了“小谢”（另案处理）、“小陈”以及“小陈”的一位朋友前来，\
               陈某用拳头和巴掌朝钟某头部打了两三下，钟某便用手蒙住头蹲在地上，陈某对钟某拳打脚踢，“小肖”到楼道里拿了一根空心铁管、\
               “小谢”拿了一根木棍朝钟某打去，之后陈某将“小肖”、“小谢”拉开，被害人钟某离开。经鉴定，钟某的损伤程度为轻伤二级。\
               公诉机关认为，应当分别以××、××追究被告人陈某的刑事责任。建议对其犯××判处一年六个月至二年六个月××、\
               犯××判处一年六个月至二年六个月××。公诉机关为支持其指控相向法庭提供了相关证据。'
               ]
    print(predictor.predict(content))

