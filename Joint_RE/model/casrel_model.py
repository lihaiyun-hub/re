"""

@-*- coding: utf-8 -*-
@ 创建人员：lhy
@ 创建时间：2025-03-22

"""
import torch
# coding:utf-8
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel

from config import *

conf = Config()


def loss(pred, gold, mask):
    pred = pred.squeeze(-1)
    los = nn.BCELoss(reduction='none')(pred, gold)
    los = torch.sum(los * mask) / torch.sum(mask)
    return los


class CasRel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.bert = BertModel.from_pretrained(conf.bert_path)
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, input_ids, mask):
        encoded_text = self.bert(input_ids, mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        pre_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pre_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pre_sub_heads, pre_sub_tails

    def get_objs_for_specific_sub(self, sub_heads2tail, sub_len, encoded_text):
        sub = torch.matmul(sub_heads2tail, encoded_text)
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pre_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pre_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        encoded_text = self.get_encoded_text(input_ids, mask)
        pre_sub_heads, pre_sub_tails = self.get_subs(encoded_text)
        sub_head2tail = sub_head2tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)
        result_dict = {'pred_sub_heads': pre_sub_heads,
                       'pred_sub_tails': pre_sub_tails,
                       'pred_obj_heads': pred_obj_heads,
                       'pred_obj_tails': pre_obj_tails,
                       'mask': mask}
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        """
        计算损失
        :param pred_sub_heads:[16, 200, 1]
        :param pred_sub_tails:[16, 200, 1]
        :param pred_obj_heads:[16, 200, 18]
        :param pred_obj_tails:[16, 200, 18]
        :param mask: shape-->[16, 200]
        :param sub_heads: shape-->[16, 200]
        :param sub_tails: shape-->[16, 200]
        :param obj_heads: shape-->[16, 200, 18]
        :param obj_tails: shape-->[16, 200, 18]
        :return:
        """
        # sub_heads.shape,sub_tails.shape, mask-->[16, 200]
        # obj_heads.shape,obj_tails.shape-->[16, 200, 18]
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        loss_1 = loss(pred_sub_heads, sub_heads, mask)
        loss_2 = loss(pred_sub_tails, sub_tails, mask)
        loss_3 = loss(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = loss(pred_obj_tails, obj_tails, rel_mask)
        return loss_1 + loss_2 + loss_3 + loss_4


def load_model(conf):
    device = conf.device
    model = CasRel(conf)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=conf.learning_rate, eps=10e-8)
    sheduler = None
    return model, optimizer, sheduler, device


if __name__ == '__main__':
    # model = CasRel(conf).to(conf.device)
    model, optimizer, sheduler, device = load_model(conf)
