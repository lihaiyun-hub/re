"""

@-*- coding: utf-8 -*-
@ 创建人员：lhy
@ 创建时间：2025-03-21

"""
import os

# 导入必备的工具包
import torch
# 导入Vocabulary，目的：用于构建, 存储和使用 `str` 到 `int` 的一一映射
from transformers import BertTokenizer, BertConfig
from fastNLP import Vocabulary
import json


# 构建配置文件Config类
class Config(object):
    def __init__(self):
        # 设置是否使用GPU来进行模型训练
        DIR_BASE = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.bert_path = os.path.join(DIR_BASE, 'data', 'bert_pretrain')
        self.bert_path = 'google-bert/bert-base-chinese'
        self.num_rel = 18  # 关系的种类数
        self.batch_size = 8
        self.train_data_path = os.path.join(DIR_BASE, 'data', 'train.json')
        self.dev_data_path = os.path.join(DIR_BASE, 'data', 'dev.json')
        self.test_data_path = os.path.join(DIR_BASE, 'data', 'test.json')
        self.rel_dict_path = os.path.join(DIR_BASE, 'data', 'relation.json')
        self.save_path = os.path.join(DIR_BASE, 'save_model')
        id2rel = json.load(open(self.rel_dict_path, encoding='utf8'))
        self.rel_vocab = Vocabulary(padding=None, unknown=None)
        # vocab更新自己的字典，输入为list列表
        self.rel_vocab.add_word_lst(list(id2rel.values()))
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # self.bert_config = BertConfig.from_pretrained(os.path.join(self.bert_path, 'bert_config.json'))
        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 20


if __name__ == '__main__':
    conf = Config()
    print(conf.rel2id)
