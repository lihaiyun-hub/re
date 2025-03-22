"""

@-*- coding: utf-8 -*-
@ 创建人员：lhy
@ 创建时间：2025-03-21

"""

from torch.utils.data import Dataset, DataLoader

from Joint_RE.utils.process import *

conf = Config()


class ReDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.datas = [json.loads(line) for line in open(data_path, encoding='utf8')]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        content = self.datas[index]
        text = content['text']
        spo_list = content['spo_list']
        return text, spo_list


def get_data():
    # 实例化训练数据集Dataset对象
    train_data = ReDataset(conf.train_data_path)

    # 实例化验证数据集Dataset对象
    dev_data = ReDataset(conf.dev_data_path)

    # 实例化测试数据集Dataset对象
    test_data = ReDataset(conf.test_data_path)

    # 实例化训练数据集Dataloader对象
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    # 实例化验证数据集Dataloader对象
    dev_dataloader = DataLoader(dataset=dev_data,
                                batch_size=conf.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
    # 实例化测试数据集Dataloader对象
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=conf.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 drop_last=True)
    return train_dataloader, dev_dataloader, test_dataloader


if __name__ == '__main__':
    rd = ReDataset(conf.train_data_path)
    train_dataloader = DataLoader(dataset=rd,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    for _ in train_dataloader:
        break
