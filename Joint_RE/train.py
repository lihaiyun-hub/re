"""

@-*- coding: utf-8 -*-
@ 创建人员：lhy
@ 创建时间：2025-03-22

"""
import os

# coding:utf-8

import pandas as pd
from tqdm import tqdm

from model.casrel_model import *
from utils.data_loader import *


def model2train(model, train_iter, dev_iter, optimizer, conf):
    epochs = conf.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), os.path.join(conf.save_path, 'last_model.pth'))


def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for step, (inputs, labels) in enumerate(tqdm(train_iter)):
        model.train()
        logist = model(**inputs)
        loss = model.compute_loss(**logist, **labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 500 == 0:

            torch.save(model.state_dict(),
                       os.path.join(conf.save_path, 'epoch_%s_model_%s.pth' % (epoch, step)))

            results = model2dev(model, dev_iter)
            print(results[-1])
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(), os.path.join(conf.save_path, 'best_f1.pth'))
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 step,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.item()))

    return best_triple_f1


def model2dev(model, dev_iter):
    '''
    验证模型效果
    :param model:
    :param dev_iter:
    :return:
    '''
    model.eval()
    # 定义一个df，来展示模型的指标。
    df = pd.DataFrame(columns=['TP', 'PRED', "REAL", 'p', 'r', 'f1'], index=['sub', 'triple'])
    df.fillna(0, inplace=True)
    # 将需要存储浮点数的列转换为 float 类型
    df['p'] = df['p'].astype('float64')
    df['r'] = df['r'].astype('float64')
    df['f1'] = df['f1'].astype('float64')
    for inputs, labels in tqdm(dev_iter):
        logist = model(**inputs)
        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):

            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(),
                                    pred_sub_tails[batch_index].squeeze())

            true_subs = extract_sub(sub_heads[batch_index].squeeze(),
                                    sub_tails[batch_index].squeeze())

            pred_objs = extract_obj_and_rel(pred_obj_heads[batch_index],
                                            pred_obj_tails[batch_index])

            true_objs = extract_obj_and_rel(obj_heads[batch_index],
                                            obj_tails[batch_index])

            # 更新 sub 相关统计
            df.loc['sub', 'PRED'] += len(pred_subs)
            df.loc['sub', 'REAL'] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df.loc['sub', 'TP'] += 1  # 直接通过 .loc 更新 TP 值

            # 更新 triple 相关统计
            df.loc['triple', 'PRED'] += len(pred_objs)
            df.loc['triple', 'REAL'] += len(true_objs)

            for true_obj in true_objs:
                if true_obj in pred_objs:
                    df.loc['triple', 'TP'] += 1

    # 计算 sub 的指标
    df.loc['sub', 'p'] = df.loc['sub', 'TP'] / (df.loc['sub', 'PRED'] + 1e-9)
    df.loc['sub', 'r'] = df.loc['sub', 'TP'] / (df.loc['sub', 'REAL'] + 1e-9)
    df.loc['sub', 'f1'] = 2 * df.loc['sub', 'p'] * df.loc['sub', 'r'] / (df.loc['sub', 'p'] + df.loc['sub', 'r'] + 1e-9)

    # 计算 sub 的中间变量（如果后续需要单独使用）
    sub_precision = df.loc['sub', 'TP'] / (df.loc['sub', 'PRED'] + 1e-9)
    sub_recall = df.loc['sub', 'TP'] / (df.loc['sub', 'REAL'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    # 计算 triple 的指标
    df.loc['triple', 'p'] = df.loc['triple', 'TP'] / (df.loc['triple', 'PRED'] + 1e-9)
    df.loc['triple', 'r'] = df.loc['triple', 'TP'] / (df.loc['triple', 'REAL'] + 1e-9)
    df.loc['triple', 'f1'] = 2 * df.loc['triple', 'p'] * df.loc['triple', 'r'] / (
                df.loc['triple', 'p'] + df.loc['triple', 'r'] + 1e-9)

    # 计算 triple 的中间变量（如果后续需要单独使用）
    triple_precision = df.loc['triple', 'TP'] / (df.loc['triple', 'PRED'] + 1e-9)
    triple_recall = df.loc['triple', 'TP'] / (df.loc['triple', 'REAL'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (triple_precision + triple_recall + 1e-9)

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    model, optimizer, sheduler, device = load_model(conf)
    train_dataloader, dev_dataloader, _ = get_data()
    model2train(model, train_dataloader, dev_dataloader, optimizer, conf)
