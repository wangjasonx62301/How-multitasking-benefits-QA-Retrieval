import pandas as pd
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset

def collate_batch(sample): #sample is List
    input_ids_batch = [s[0] for s in sample]
    mask_batch = [s[1] for s in sample]
    token_batch = [s[2] for s in sample]
    label_batch = [s[3] for s in sample]

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True)
    mask_batch = pad_sequence(mask_batch, batch_first=True)
    token_batch = pad_sequence(token_batch, batch_first=True)
    #label_batch = pad_sequence(label_batch, batch_first=True)
    label_batch = torch.tensor(label_batch, dtype=torch.float)

    return input_ids_batch, mask_batch, token_batch, label_batch 

class Classification_dataset(Dataset):
    def __init__(self, df, tokenizer, num_labels):
        self.df = df
        self.tokenizer = tokenizer
        self.num_labels = num_labels
    
    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])
        label = [0.] * self.num_labels
        label[df['label'][index]] = 1.

        return input_ids, mask, token, label
    
    def __len__(self):
        return len(self.df)
    