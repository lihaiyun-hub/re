# coding:utf-8
import json
from torch.utils.data import DataLoader, Dataset
from utils.process import *
conf = Config()

class ReDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.datas = [json.loads(line) for line in open(data_path, encoding='utf-8')]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # 根据索引获取某个样本
        content = self.datas[item]
        text = content["text"]
        spo_list = content["spo_list"]
        return text, spo_list


def get_data():
    train_dataset = ReDataset(data_path=conf.train_data_path)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=True)
    dev_dataset = ReDataset(data_path=conf.dev_data_path)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=True)
    test_dataset = ReDataset(data_path=conf.test_data_path)
    test_dataloader = DataLoader(dataset=test_dataset,
                                  batch_size=conf.batch_size,
                                  collate_fn=collate_fn,
                                  drop_last=True,
                                  shuffle=True)
    return train_dataloader, dev_dataloader, test_dataloader
if __name__ == '__main__':
    train_dataloader, dev_dataloader, test_dataloader = get_data()
    for inputs, labels in train_dataloader:
        print(f'inputs-->{inputs}')
        print(f'inputs["input_ids"]-->{inputs["input_ids"].shape}')
        print(f'inputs["mask"]-->{inputs["mask"].shape}')
        print(f'inputs["sub_head2tail"]-->{inputs["sub_head2tail"].shape}')
        print(f'labels-->{labels}')
        print(f'labels["sub_heads"]-->{labels["sub_heads"].shape}')
        print(f'labels["obj_heads"]-->{labels["obj_heads"].shape}')
        break