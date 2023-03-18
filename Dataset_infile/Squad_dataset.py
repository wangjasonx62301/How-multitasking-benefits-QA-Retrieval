import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
from torch.utils.data import Dataset

from transformers import BertModel
from transformers import BertTokenizer
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from typing import *


class SQUAD_Dataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context'][index], df['question'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])

        return input_ids, mask, token, df['TKstart'][index], df['TKend'][index], df['SEP_ind'][index]
    
    def __len__(self):
        return len(self.df)

def collate_batch(sample): #sample is List
    input_ids_batch = [s[0] for s in sample]
    mask_batch = [s[1] for s in sample]
    token_batch = [s[2] for s in sample]
    Start_batch = [s[3] for s in sample]
    End_batch = [s[4] for s in sample]
    SEP_index_batch = [s[5] for s in sample]

    input_ids_batch = pad_sequence(input_ids_batch, batch_first=True)
    mask_batch = pad_sequence(mask_batch, batch_first=True)
    token_batch = pad_sequence(token_batch, batch_first=True)

    return input_ids_batch, mask_batch, token_batch, SEP_index_batch, Start_batch, End_batch


class Squad_Loader(object):
    def __init__(self):
        df = pd.read_csv('nlp\\Dataset_infile\\SQUAD_train.csv')
        self.dataset = SQUAD_Dataset(df)
        self.BATCH_SIZE = 2

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# CoLA_ = CoLA_Loader()
# train_loader = CoLA_.get_loader()

# data = next(iter(train_loader))
# print(data)
# print(train_loader)

    
