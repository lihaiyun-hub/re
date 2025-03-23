# coding:utf-8
import os.path

from model.CasrelModel import *
from utils.process import *
from utils.data_loader import *
from config import *
import pandas as pd
from tqdm import tqdm


# 定义主训练方法
def mode2train(model, train_iter, dev_iter, optimizer, conf):
    epochs = conf.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        best_triple_f1 = train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch)
    torch.save(model.state_dict(), os.path.join(conf.save_path,'last_model.pth'))


def train_epoch(model, train_iter, dev_iter, optimizer, best_triple_f1, epoch):
    for step, (inputs, labels) in enumerate(tqdm(train_iter, desc='Casrel模型训练')):
        model.train()
        # 得到模型的预测结果
        logits = model(**inputs)
        # 计算损失
        loss = model.compute_loss(**logits, **labels)
        # 梯度清零
        optimizer.zero_grad()
        # model.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        if (step + 1) % 500 == 0:
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
    # 将df里面的值全部赋值为0
    df.fillna(0, inplace=True)
    # print(f'df-->{df}')
    for inputs, labels in tqdm(dev_iter, desc='Casrel模型验证'):
        logist = model(**inputs)
        # logist['pred_sub_heads']-->shape->[4, 70, 1]
        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])
        # 针对每个样本去预测结果
        for batch_idx in range(batch_size):
            pred_subs = extract_sub(pred_sub_heads[batch_idx].squeeze(),
                                    pred_sub_tails[batch_idx].squeeze())
            # print(f'pred_subs--》{pred_subs}')
            true_subs = extract_sub(sub_heads[batch_idx].squeeze(),
                                    sub_tails[batch_idx].squeeze())
            # print(f'true_subs--》{true_subs}')
            pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_idx],
                                            pred_obj_tails[batch_idx])
            # print(f'pred_ojbs--》{pred_ojbs}')
            true_objs = extract_obj_and_rel(obj_heads[batch_idx],
                                            obj_tails[batch_idx])
            # print(f'true_objs--》{true_objs}')
            # df["PRED"]["sub"] += len(pred_subs)
            df.loc["sub", "PRED"] += len(pred_subs)
            # df["REAL"]["sub"] += len(true_subs)
            df.loc["sub", "REAL"] += len(true_subs)

            for true_sub in true_subs:
                if true_sub in pred_subs:
                    # df["TP"]["sub"] += 1
                    df.loc['sub', 'TP'] += 1

            # df["PRED"]["triple"] += len(pred_ojbs)
            df.loc["triple", "PRED"] += len(pred_ojbs)
            # df["REAL"]["triple"] += len(true_objs)
            df.loc["triple", "REAL"] += len(true_objs)

            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    # df["TP"]["triple"] += 1
                    df.loc["triple", 'TP'] += 1
    # 计算指标
    # 计算主实体的p、r、f1
    # 计算主实体的precision
    df.loc["sub", "p"] = df.loc['sub', 'TP'] / (df.loc["sub", "PRED"] + 1e-9)
    # 计算主实体的recall
    df.loc["sub", "r"] = df.loc['sub', 'TP'] / (df.loc["sub", "REAL"] + 1e-9)
    # 计算主实体的F1
    df.loc["sub", 'f1'] = 2 * df.loc["sub", "p"] * df.loc["sub", "r"] / (df.loc["sub", "p"] + df.loc["sub", "r"] + 1e-9)

    sub_precision = df.loc["sub", "p"]
    sub_recall = df.loc["sub", "r"]
    sub_f1 = df.loc["sub", 'f1']
    # 计算客实体及关系的p、r、f1
    # 计算客实体的precision
    df.loc["triple", "p"] = df.loc['triple', 'TP'] / (df.loc["triple", "PRED"] + 1e-9)
    # 计算主实体的recall
    df.loc["triple", "r"] = df.loc['triple', 'TP'] / (df.loc["triple", "REAL"] + 1e-9)
    # 计算主实体的F1
    df.loc["triple", 'f1'] = 2 * df.loc["triple", "p"] * df.loc["triple", "r"] / (
                df.loc["triple", "p"] + df.loc["triple", "r"] + 1e-9)

    triple_precision = df.loc["triple", "p"]
    triple_recall = df.loc["triple", "r"]
    triple_f1 = df.loc["triple", 'f1']

    return sub_precision, sub_recall, sub_f1, triple_precision, triple_recall, triple_f1, df


if __name__ == '__main__':
    conf = Config()
    model, optimizer, sheduler, device = load_model(conf)
    train_iter, dev_iter, _ = get_data()
    mode2train(model, train_iter, dev_iter, optimizer, conf)
