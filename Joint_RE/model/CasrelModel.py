# coding:utf-8
import torch
import torch.nn as nn
from transformers import BertModel
import torch.optim as optim
from config import *
from utils.data_loader import *

conf = Config()
# 定义Casrel模型类
class Casrel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        # 加载bert预训练模型
        self.bert = BertModel.from_pretrained(conf.bert_path)
        # 定义第一个全连接层：识别主实体的开始位置
        self.sub_heads_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第二个全连接层：识别主实体的结束位置
        self.sub_tails_linear = nn.Linear(conf.bert_dim, 1)
        # 定义第三个全连接层：识别客实体的开始位置及对应的关系
        self.obj_heads_linear = nn.Linear(conf.bert_dim, conf.num_rel)
        # 定义第四个全连接层：识别客实体的结束位置及对应的关系
        self.obj_tails_linear = nn.Linear(conf.bert_dim, conf.num_rel)

    def get_encoded_text(self, input_ids, mask):
         # 得到bert模型编码之后的结果
        encoded_text = self.bert(input_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        # 预测主实体的开始位置
        sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return sub_heads, sub_tails

    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        '''
        将subject实体信息融合原始句子中：将主实体字向量实现平均，然后加在当前句子的每一个字向量上，进行计算
        :param sub_head2tail:shape-->【4，1, 67】
        :param sub_len:shape--->[4,1]
        :param encoded_text:.shape[4，67，768]
        :return:
            pred_obj_heads-->shape []
            pre_obj_tails-->shape []
        '''
        # 筛选出主实体的bert编码后的信息 # sub-->shape-->[4, 1, 768]
        sub = torch.matmul(sub_head2tail, encoded_text)
        # 需要将上一步骤sub主实体的信息进行平均:sub_avg-->[4, 1, 768]
        sub_len = sub_len.unsqueeze(1)
        sub_avg = sub / sub_len
        # 将平均之后的sub信息和原始的bert编码后的信息进行融合
        encoded_text = sub_avg + encoded_text # [4, 67, 768]
        # 预测客实体的开始位置及关系:obj_heads--shape->[4, 67, 18]
        obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        # 预测客实体的结束位置及关系:obj_tails--shape->[4, 67, 18]
        obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return obj_heads, obj_tails

    def forward(self,input_ids, mask, sub_head2tail, sub_len):
        # input_ids, mask, sub_head2tail--》shape-->[4, 67];sub_len-->[4, 1]
        # 1.得到bert模型编码之后的结果:encoded_text-->shape-->[4, 67, 768]
        encoded_text = self.get_encoded_text(input_ids, mask)
        # print(f'bert编码之后的结果--》{encoded_text.shape}')
        # 2.基于bert模型编码之后的结果，预测主实体的开始和结束位置
        # 2.1 sub_heads_pred-->sub_tails_pred-->shape-->[4, 67, 1]
        sub_heads_pred,  sub_tails_pred = self.get_subs(encoded_text)

        # 3. 基于bert模型编码之后的结果+选择的主实体的信息，预测客实体的开始和结束位置（包含关系）
        sub_head2tail = sub_head2tail.unsqueeze(1) # [4, 1, 67]
        obj_heads_pred, obj_tails_pred = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)

        result_dict = {'pred_sub_heads': sub_heads_pred,
                       'pred_sub_tails': sub_tails_pred,
                       'pred_obj_heads': obj_heads_pred,
                       'pred_obj_tails': obj_tails_pred,
                       'mask': mask}
        return result_dict

    def compute_loss(self,
                     pred_sub_heads, pred_sub_tails,
                     pred_obj_heads, pred_obj_tails,
                     mask,
                     sub_heads, sub_tails,
                     obj_heads, obj_tails):
        '''
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
        '''
        # sub_heads.shape,sub_tails.shape, mask-->[16, 200]
        # obj_heads.shape,obj_tails.shape-->[16, 200, 18]
        # 1. 获取关系类别的总数
        rel_count = obj_heads.size(-1)
        # 2.对mask矩阵进行repeat: -->rel_mask-->[16, 200, 18]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        # 3. 计算主实体开始位置预测的损失值
        loss1 = self.loss(pred_sub_heads, sub_heads, mask)
        # # 4. 计算主实体结束位置预测的损失值
        loss2 = self.loss(pred_sub_tails, sub_tails, mask)
        # 5. 计算客实体开始位置及关系预测的损失值
        loss3 = self.loss(pred_obj_heads, obj_heads, rel_mask)
        # print(f'loss3--》{loss3}')
        # 6. 计算客实体结束位置及关系预测的损失值
        loss4 = self.loss(pred_obj_tails, obj_tails, rel_mask)

        return loss1+loss2+loss3+loss4

    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        # print(f'gold--》真实结果的shape-->{gold.shape}')
        # 使用BCEloss
        tmp_loss = nn.BCELoss(reduction='none')(pred, gold)
        # print(f'tmp_loss--》{tmp_loss}')
        # print(f'tmp_loss--》{tmp_loss.shape}')
        # 计算平均损失（去除padding的影响）
        los = torch.sum(tmp_loss*mask) / torch.sum(mask)
        return los

def load_model(conf):
    device = conf.device
    model = Casrel(conf)
    model.to(device)
    # 因为本次模型借助BERT做fine_tuning， 因此需要对模型中的大部分参数进行L2正则处理防止过拟合，包括权重w和偏置b
    # prepare optimzier
    # named_parameters()获取模型中的参数和参数名字
    print(f'model_-->{model.parameters()}')
    print(f'model_type-->{type(model.parameters())}')
    param_optimizer = list(model.named_parameters())
    # print(f'param_optimizer--->{param_optimizer}')
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]  # no_decay中存放不进行权重衰减的参数{因为bert官方代码对这三项免于正则化}
    # any()函数用于判断给定的可迭代参数iterable是否全部为False，则返回False，如果有一个为True，则返回True
    # 判断param_optimizer中所有的参数。如果不在no_decay中，则进行权重衰减;如果在no_decay中，则不进行权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=conf.learning_rate)
    # 是否需要对bert进行warm_up。这里默认不进行
    sheduler = None

    return model, optimizer, sheduler, device
if __name__ == '__main__':
    # model = Casrel(conf)
    # print(model)
    # train_dataloader, dev_dataloader, test_dataloader = get_data()
    # for inputs, labels in train_dataloader:
    #     results = model(**inputs)
    #     my_loss = model.compute_loss(**results, **labels)
    #     print(f'my_loss--》{my_loss}')
    #     # print(f'pred_sub_heads--》{results["pred_sub_heads"].shape}')
    #     # print(f'pred_sub_tails--》{results["pred_sub_tails"].shape}')
    #     # print(f'pred_obj_heads--》{results["pred_obj_heads"].shape}')
    #     # print(f'pred_obj_tails--》{results["pred_obj_tails"].shape}')
    #     break
    load_model(conf)



