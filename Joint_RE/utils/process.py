# coding:utf-8
from config import *
import torch
from random import choice
from collections import defaultdict
conf = Config()
from pprint import pprint

def find_head_idx(source, target):
    # # 获取实体的开始索引位置
    len_target = len(target)
    for i in range(len(source)):
        if source[i: i+len_target] == target:
            return i
    return -1


def create_label(inner_triples, inner_input_ids, seq_len):
    # 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
    inner_sub_heads, inner_sub_tails = torch.zeros(seq_len), torch.zeros(seq_len)
    # 我是张三--》[0, 1, 2, 3]-->heads-->[0, 0,0,0]-->[0, 0, 1,0];tails-->[0, 0, 0, 1]
    inner_obj_heads = torch.zeros((seq_len, conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len, conf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)
    inner_sub_len = torch.tensor([1], dtype=torch.float)
    # 主词到谓词的映射
    s2ro_map = defaultdict(list)
    for inner_triple in inner_triples:
        # print(f'inner_triple--》{inner_triple}')
        sub = conf.tokenizer(inner_triple["subject"], add_special_tokens=False)['input_ids']
        predict = conf.rel_vocab.to_index(inner_triple["predicate"])
        obj = conf.tokenizer(inner_triple["object"], add_special_tokens=False)['input_ids']
        inner_triple = (sub, predict, obj)
        # print(f'inner_triple==》{inner_triple}')
        sub_head_idx = find_head_idx(inner_input_ids, inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids, inner_triple[2])
        if sub_head_idx != -1 and obj_head_idx != -1:
            # sub(开始索引位置，结束索引位置)
            sub = (sub_head_idx, sub_head_idx+len(inner_triple[0])-1)
            # print(f'sub-->{sub}')
            s2ro_map[sub].append((obj_head_idx, obj_head_idx+len(inner_triple[2])-1, inner_triple[1]))

    # print(f's2ro_map--》{s2ro_map}')
    if s2ro_map:
        for sub in s2ro_map.keys():
            inner_sub_heads[sub[0]] = 1
            inner_sub_tails[sub[1]] = 1
        # print(f'inner_sub_heads---》{inner_sub_heads}')
        # print(f'inner_sub_tails---》{inner_sub_tails}')
        sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
        inner_sub_head2tail[sub_head_idx: sub_tail_idx+1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx+1-sub_head_idx], dtype=torch.float)
        for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
            inner_obj_heads[ro[0]][ro[2]] = 1
            inner_obj_tails[ro[1]][ro[2]] = 1
    return inner_sub_len, inner_sub_head2tail, inner_sub_heads, inner_sub_tails, inner_obj_heads, inner_obj_tails


def collate_fn(batch):
    # print(f'batch--》{batch}')
    text_list = [data[0] for data in batch]
    triple = [data[1] for data in batch]
    # print(f'text_list--》{text_list}')
    # print(f'triple--》{triple}')
    # 按照一个批次中的最长句子进行补齐
    text = conf.tokenizer.batch_encode_plus(text_list, padding=True)
    # pprint(f'text-->{text}')
    batch_size = len(text["input_ids"])
    seq_len = len(text["input_ids"][0])
    sub_heads = []
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []
    sub_head2tail = []
    # 循环遍历每个样本，将实体信息进行张量的转化
    for batch_index in range(batch_size):
        # 根据索引获得一个样本的input_ids
        inner_input_ids = text["input_ids"][batch_index]
        # print(f'inner_input_ids--》{inner_input_ids}')
        #  根据索引获得一个样本的对应的spo_list
        inner_triples = triple[batch_index]
        # print(f'inner_triples--{inner_triples}')
        # 获取每个样本的：主实体长度、主实体开始和结束位置张量表示、客实体以及对应关系实现张量表示
        results = create_label(inner_triples, inner_input_ids, seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])

    input_ids = torch.tensor(text["input_ids"], device=conf.device)
    mask = torch.tensor(text["attention_mask"], device=conf.device)
    # 借助torch.stack()函数沿一个新维度对输入batch_size张量序列进行连接，序列中所有张量应为相同形状；stack 函数返回的结果会新增一个维度,
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)

    inputs = {
        'input_ids': input_ids,
        'mask': mask,
        'sub_head2tail': sub_head2tail,
        'sub_len': sub_len
    }
    labels = {
        'sub_heads': sub_heads,
        'sub_tails': sub_tails,
        'obj_heads': obj_heads,
        'obj_tails': obj_tails
    }

    return inputs, labels


def extract_sub(pred_sub_heads, pred_sub_tails):
    '''
    :param pred_sub_heads: 模型预测出的主实体开头位置-->shape[sequence_length]-->[70]
    :param pred_sub_tails: 模型预测出的主实体尾部位置-->shape[sequence_length]-->[70]
    :return: subs列表里面对应的所有实体【head, tail】
    '''
    subs = []
    # 找出值为1的索引
    heads = torch.arange(0, len(pred_sub_heads), device=conf.device)[pred_sub_heads == 1]
    tails = torch.arange(0, len(pred_sub_tails), device=conf.device)[pred_sub_tails == 1]
    for head, tail in zip(heads, tails):
        if tail >= head:
            subs.append((head.item(), tail.item()))
    return subs


def extract_obj_and_rel(obj_heads, obj_tails):
    '''

    :param obj_heads:  模型预测出的从实体开头位置以及关系类型-->shape-->[seq_len, num_rel]-->[70, 18]
    :param obj_tails:  模型预测出的从实体尾部位置以及关系类型-->shape-->[seq_len, num_rel]-->[70, 18]
    :return: obj_and_rels：元素形状：(rel_index, start_index, end_index)
    '''
    # print(f'obj_heads--》{obj_heads.shape}')
    # print(f'obj_tails--》{obj_tails.shape}')
    obj_heads = obj_heads.T # [num_rel, seq_len]-->[18, 70]
    obj_tails = obj_tails.T # [num_rel, seq_len]-->[18, 70]
    rel_count = obj_heads.shape[0]
    obj_and_rels = []

    # 对每种关系去预测客实体的开始和结束的位置
    for rel_index in range(rel_count):
        obj_head = obj_heads[rel_index] # [70]
        obj_tail = obj_tails[rel_index] # [70]
        objs = extract_sub(obj_head, obj_tail)
        if objs:
            for obj in objs:
                start_idx, end_idx = obj
                obj_and_rels.append((rel_index, start_idx, end_idx))

    return obj_and_rels




def convert_score_to_zero_one(tensor):
    '''
    以0.5为阈值，大于0.5的设置为1，小于0.5的设置为0
    '''
    tensor[tensor >= 0.5] = 1
    tensor[tensor < 0.5] = 0
    return tensor