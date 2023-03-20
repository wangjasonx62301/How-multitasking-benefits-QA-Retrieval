from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from typing import *

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


def pading_empty_tensor(context_LHL):  #context(QA) last hidden layer, this is padding for lstm because the context length may be difference
    seqlen = [s.size(0) for s in context_LHL]
    data = pack_padded_sequence(context_LHL, seqlen, batch_first=True, enforce_sorted=False)
    return data


def get_retrieve_context_matrix(SEPind:List, seq_len, hidden_layer_size): #last hidden layer
    num = len(SEPind)  #通常是 batch size
    
    AOLN = [1] * hidden_layer_size  # a_output_hidden_layer_needed
    AOLD = [0] * hidden_layer_size  # a_output_hidden_DONT_needed

    mtx = [[AOLN]*SEPind[i] + [AOLD]*(seq_len-SEPind[i]) for i in range(num)]
    mtx = torch.tensor(mtx, requires_grad=False)
    
    return mtx
    

# create dataset
class QAdataset(Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.df = df
    
    def __getitem__(self, index):
        df = self.df
        EC = self.tokenizer.encode_plus(df['context'][index], df['question'][index])
        
        input_ids = torch.tensor(EC['input_ids'])
        mask = torch.tensor(EC['attention_mask'])
        token = torch.tensor(EC['token_type_ids'])

        return input_ids, mask, token, df['TKstart'][index], df['TKend'][index], df['SEP_ind'][index]
    
    def __len__(self):
        return len(self.df)
    