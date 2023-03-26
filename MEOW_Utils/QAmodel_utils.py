from torch.nn.utils.rnn import pack_padded_sequence
import torch

from typing import *


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
    


