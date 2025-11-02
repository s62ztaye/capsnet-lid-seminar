# preprocess.py
import pandas as pd
from torch.utils.data import Dataset
import torch

class LIDataset(Dataset):
    def __init__(self, filepath='wili-2018.txt', lang_subset=50, max_len=200):
        df = pd.read_csv(filepath, sep='§§', engine='python', names=['text', 'label'])
        top_langs = df['label'].value_counts().index[:lang_subset]
        df = df[df['label'].isin(top_langs)].reset_index(drop=True)
        
        self.texts = df['text'].str.lower().apply(lambda s: s[:max_len]).tolist()
        self.labels = pd.Categorical(df['label']).codes.tolist()
        self.char2idx = {chr(i): i - 32 for i in range(32, 127)}  # ASCII 32-126
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char2idx.get(c, 0) for c in text]
        padding = [0] * (self.max_len - len(indices))
        indices = indices + padding
        return torch.tensor(indices), torch.tensor(self.labels[idx])
