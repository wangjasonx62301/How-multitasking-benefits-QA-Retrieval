import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
input_file_path = r'/kaggle/input/squad-2/train-v2.0.json'


# Bert_QA

from transformers import BertModel
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def pading_empty_tensor(context_LHL):  #last hidden layer, this is padding for lstm because the context length may be difference
    context_LHL = pad_sequence(context_LHL, batch_first=True)

    seqlen = [s.size(0) for s in context_LHL]
    data = pack_padded_sequence(context_LHL, seqlen, batch_first=True, enforce_sorted=False)

    return data

class Bert_QA(torch.nn.Module):
    def __init__(self, model, embedding_layer):
        super(Bert_QA, self).__init__()
        
        self.model = model
        self.embedding_layer = embedding_layer

        self.LSTM = torch.nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)

        self.start_clasifier = torch.tensor([1.]*768).to(device)
        self.end_clasifier = torch.tensor([1.]*768).to(device)

        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    # return 出(batch, seq_len)
    def forward(self, input_ids, attention_mask, token, SEPind, start_pos, end_pos):
        # last hidden layer
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        outputs = self.model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
        last_hidden_layer = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size:768)

        total_loss = 0
        correct_num = 0
        
        context_LHL = [last_hidden_layer[i][0:SEPind[i]] for i in range(BATCH_SIZE)] # 現在是一個tensor的list
        
        context_PACk = pading_empty_tensor(context_LHL) # (batch_size, context_padding_length, 768)
        
        output, (hn,cn) = self.LSTM(context_PACk)
        output, input_sizes = pad_packed_sequence(output, batch_first=True) # output is (batch_size, context_padding_length, 768)
        

        start_score = (output * self.start_clasifier).sum(dim=2) # (batch_size, context_padding_length)
        end_score = (output * self.end_clasifier).sum(dim=2) # (batch_size, context_padding_length)
    
        start_1hot = torch.zeros(BATCH_SIZE, output.size(1)).to(device)
        end_1hot = torch.zeros(BATCH_SIZE, output.size(1)).to(device)
        
        for i in range(BATCH_SIZE):
            start_1hot[i][start_pos[i]] = 1
            end_1hot[i][end_pos[i]] = 1

        loss_start = self.loss_function(start_score, start_1hot)# (batch_size, context_padding_length)
        loss_end = self.loss_function(end_score, end_1hot)
        
        
        total_loss = loss_start + loss_end
        return total_loss

        
        # # 不須經過 lstm
        # for i in range(len(last_hidden_layer)):
        #     QLHL = last_hidden_layer[i][0:SEPind[i]]  # (context_length, 768)
        #     s1hot = start_1hot[i][0:SEPind[i]]  # (context_length)
        #     e1hot = end_1hot[i][0:SEPind[i]]  # (context_length)

        #     start_score = (QLHL * self.start_clasifier).sum(dim=1) # (context_length)
        #     end_score = (QLHL * self.end_clasifier).sum(dim=1)
            
        #     # start_prob = self.softmax(start_score) # torch.nn.CrossEntropyLoss 已經有softmax ,這行只是方便觀察
        #     # end_prob = self.softmax(end_score)

        #     # loss_start = self.loss_function(start_prob, s1hot)
        #     # loss_end = self.loss_function(end_prob, e1hot)

        #     loss_start = self.loss_function(start_score, s1hot)
        #     loss_end = self.loss_function(end_score, e1hot)
            
        #     loss = loss_start + loss_end
        #     total_loss += loss
        #     correct_num += 1 if s1hot[start_score.argmax()]==1 and e1hot[end_score.argmax()]==1
        
        return total_loss, correct_num