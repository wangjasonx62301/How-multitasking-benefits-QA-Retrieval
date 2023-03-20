# Bert_QA

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from MEOW_Models.MT_models import*
from MEOW_Utils.QA_utils import*

## embedding_layer is the embedding_layer from bert

class Bert_QA(torch.nn.Module):
    def __init__(self, model:BertWithoutEmbedding, embedding_layer, device):
        super(Bert_QA, self).__init__()
        
        self.device = device
        self.model = model  
        self.embedding_layer = embedding_layer

        self.LSTM = torch.nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True)

        self.start_clasifier = torch.tensor([1.]*768).to(self.device)
        self.end_clasifier = torch.tensor([1.]*768).to(self.device)

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
        
        mtx = get_retrieve_context_matrix(SEPind, last_hidden_layer.size(1), last_hidden_layer.size(2))
        mtx = mtx.to(self.device)

        context_LHL = last_hidden_layer * mtx   #last hidden layer
        
        # 錯誤寫法
        # context_LHL = [last_hidden_layer[i][0:SEPind[i]] for i in range(BATCH_SIZE)] # 現在是一個tensor的list
        
        context_PACk = pading_empty_tensor(context_LHL) # (batch_size, context_padding_length, 768)
        
        output, (hn,cn) = self.LSTM(context_PACk)
        output, input_sizes = pad_packed_sequence(output, batch_first=True) # output is (batch_size, context_padding_length, 768)
        

        start_score = (output * self.start_clasifier).sum(dim=2) # (batch_size, context_padding_length)
        end_score = (output * self.end_clasifier).sum(dim=2) # (batch_size, context_padding_length)
    
        this_batch_size = input_ids.size(0)
        start_1hot = torch.zeros(this_batch_size, output.size(1)).to(self.device)
        end_1hot = torch.zeros(this_batch_size, output.size(1)).to(self.device)
        
        for i in range(this_batch_size):
            start_1hot[i][start_pos[i]] = 1
            end_1hot[i][end_pos[i]] = 1

        loss_start = self.loss_function(start_score, start_1hot)# (batch_size, context_padding_length)
        loss_end = self.loss_function(end_score, end_1hot)
        
        
        total_loss = loss_start + loss_end
        return total_loss