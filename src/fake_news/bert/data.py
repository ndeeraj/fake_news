import pandas as pd
import torch
from fake_news.bert import constants
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split


def train_test():
    dfs = []

    for fn in constants.data_fn:
        p = Path(constants.data_dir, fn)
        _df = pd.read_csv(p)
        cls = 0
        if 'Fake' in fn:
            cls = 1
        _df['class'] = cls
        dfs.append(_df)

    df = pd.concat(dfs)

    train, test = train_test_split(df, random_state=constants.seed, shuffle=True, test_size=constants.test_pct)
    return train, test


class FakeNewsDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        rec = self.data.iloc[item]
        inp = f"{rec['title']}: {rec['text']}"
        cls = rec['class']
        return inp, cls

    def collator(self, batch):
        inp, cls = zip(*batch)
        b = self.tokenizer(inp, padding=True, truncation=True, max_length=512, return_tensors='pt')
        b['labels'] = torch.LongTensor(cls)
        return b

