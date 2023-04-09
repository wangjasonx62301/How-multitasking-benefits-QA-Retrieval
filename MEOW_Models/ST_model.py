# there are three module for three different task
# single-seq-classification, pairwise-seq-classification and QA-retrivel

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from MEOW_Models.Kernel_model import BertWithoutEmbedding, Modeling
from typing import*

from transformers.models.bert.modeling_bert import BertEmbeddings
from MEOW_Utils.model_utils import get_retrieve_context_matrix, pading_empty_tensor, Highway_layer, CLS_pooler_layer
from torch.nn.parameter import Parameter
import random

## one sequence claassification 是由 [CLS] 的 contextualize embedding 完之結果再進行 classifier

class Bert_classification(torch.nn.Module):
    def __init__(
        self,
        model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        device,
        num_labels : int
        ):
        super(Bert_classification, self).__init__()
        
        self.device = device
        self.model = model
        self.embedding_layer = embedding_layer
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
        
        self.loss_function = torch.nn.CrossEntropyLoss() #會自己 softmax
        self.softmax = torch.nn.Softmax(dim=1)
        
        
    # return 出(batch, seq_len)
    def forward(
        self, 
        input_ids : torch.tensor, 
        attention_mask : torch.tensor, 
        token : torch.tensor, 
        label : torch.tensor, 
        SEPind : List 
        ) -> Tuple[torch.tensor]: #loss and probability

        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        outputs = self.model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
    
        last_hidden_layer = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size:768)

        # sep_list = [last_hidden_layer[i,SEPind[i], :] for i in range(len(SEPind))]
        sep_list = [last_hidden_layer[i,1, :] for i in range(len(SEPind))]
        for_sep = torch.stack(sep_list)

        sep_output = self.dropout(for_sep)
        logits = self.classifier(sep_output)

        loss = self.loss_function(logits, label)
        prob = self.softmax(logits)
        
        return loss, prob
    
## pairwise 是由 [CLS] 的 contextualize embedding 完之結果再進行 classifier
## 有想過改成 [SEP] 的 contextualize embedding 
## 不過因 [SEP] 在每個 train dataset 都是不同位置，不確定效能是否會變差

class Bert_pairwise(torch.nn.Module):
    def __init__(
        self,
        model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        device,
        num_labels : int
        ):
        super(Bert_pairwise, self).__init__()
        
        self.device = device
        self.model = model
        self.embedding_layer = embedding_layer
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
        
        self.loss_function = torch.nn.CrossEntropyLoss() #會自己 softmax
        self.softmax = torch.nn.Softmax(dim=1)
        
        
    # return 出(batch, seq_len)
    def forward(
        self, 
        input_ids : torch.tensor, 
        attention_mask : torch.tensor, 
        token : torch.tensor, 
        label : torch.tensor, 
        SEPind : List 
        ) -> Tuple[torch.tensor]: #loss and probability

        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        outputs = self.model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
    
        last_hidden_layer = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size:768)

        # sep_list = [last_hidden_layer[i,SEPind[i], :] for i in range(len(SEPind))]
        sep_list = [last_hidden_layer[i,1, :] for i in range(len(SEPind))]
        for_sep = torch.stack(sep_list)

        sep_output = self.dropout(for_sep)
        logits = self.classifier(sep_output)

        loss = self.loss_function(logits, label)
        prob = self.softmax(logits)
        
        return loss, prob

## embedding_layer is the embedding_layer from bert
## QA 層的輸出是用 context 的 contextualize embedding 完之結果再經過 LSTM
## 仍可修改 , LSTM 理想狀態是兩層

class Bert_QA(torch.nn.Module):
    def get_1d_init_tensor(self,m):
        w = [0] * m
        for i in range(m):
            w[i] = random.uniform(-0.8, 0.8)
        w = torch.tensor(w, requires_grad=True)
        return w

    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        num_labels : int,
        device
        ):
        super(Bert_QA, self).__init__()
        
        self.device = device
        self.embedding_layer = embedding_layer
        self.kernel_model = kernel_model

        self.hid_size = kernel_model.config.hidden_size
        self.modeling_layer = Modeling(hid_size = self.hid_size, device=device)

        self.start_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.end_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.CLS_pooler = CLS_pooler_layer(num_labels=num_labels)
        self.loss_function_label = torch.nn.CrossEntropyLoss()

        # self.start_clasifier = torch.nn.Linear(hid_size, hid_size)
        # self.end_clasifier = torch.nn.Linear(hid_size, hid_size)

        
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids : torch.tensor,
        attention_mask : torch.tensor,
        token : torch.tensor,
        label : torch.tensor,
        SEPind : List,
        start_pos : List,
        end_pos : List,
        return_start_end_pos : bool = False
        ) -> Tuple[torch.Tensor]: # return loss only
        
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        bert_output = self.kernel_model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
        output = self.modeling_layer(SEPind, bert_output)

        # this CLS output is not the bert output, it has been the modeling layer
        CLS_list = [output[i,0, :] for i in range(len(SEPind))]
        CLS_output = torch.stack(CLS_list)
        logits = self.CLS_pooler(CLS_output)
        loss_for_label = self.loss_function_label(logits, label)

        prob = self.softmax(logits)

        start_score = (output * self.start_clasifier).sum(dim=2) # (batch_size, context_padding_length)
        end_score = (output * self.end_clasifier).sum(dim=2) # (batch_size, context_padding_length)

        # print(start_score)
        # print(end_score)
        # print(start_score.argmax(dim=1))
        # print(end_score.argmax(dim=1))
    
        this_batch_size = input_ids.size(0)
        start_1hot = torch.zeros(this_batch_size, output.size(1)).to(self.device)
        end_1hot = torch.zeros(this_batch_size, output.size(1)).to(self.device)
        
        for i in range(this_batch_size):
            start_1hot[i][start_pos[i]] = 1
            end_1hot[i][end_pos[i]] = 1

        loss_start = self.loss_function(start_score, start_1hot)# (batch_size, context_padding_length)
        loss_end = self.loss_function(end_score, end_1hot) + loss_start
        
        loss_for_pos = loss_start + loss_end
        total_loss = loss_for_pos + loss_for_label

        if return_start_end_pos == True:
            return start_score.argmax(dim=1), end_score.argmax(dim=1)
        
        return total_loss, prob
    