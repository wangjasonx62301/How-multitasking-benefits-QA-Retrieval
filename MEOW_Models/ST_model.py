import torch
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
from typing import*

from transformers.models.bert.modeling_bert import BertEmbeddings, BertPooler
from MEOW_Utils.model_utils import get_retrieve_context_matrix, pading_empty_tensor, Highway_layer, CLS_pooler_layer
from torch.nn.parameter import Parameter
import random

class Bert_classification(torch.nn.Module):
    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        embedding_layer : BertEmbeddings,
        modeling_layer : ModelingCLF,
        num_labels : int,
        device
        ):
        super(Bert_classification, self).__init__()
        
        self.kernel_model = kernel_model
        self.embedding_layer = embedding_layer
        self.modeling_layer = modeling_layer

        self.device = device
        self.hid_size = kernel_model.config.hidden_size

        self.clf_clasifier = torch.nn.Linear(self.hid_size, num_labels)
        
        self.loss_function = torch.nn.CrossEntropyLoss() #會自己 softmax
        self.softmax = torch.nn.Softmax(dim=1)
           
    # return 出(batch, seq_len)
    def forward(
        self, 
        input_ids : torch.tensor, 
        attention_mask : torch.tensor, 
        token : torch.tensor,
        SEPind : List,
        label : torch.tensor = None
        ) -> Tuple[torch.tensor]: #loss and probability

        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        bert_output = self.kernel_model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token)

        output_clf = self.modeling_layer(SEPind, bert_output)  # (batch_size, 768) 
        clf_score = self.clf_clasifier(output_clf) # (batch_size, 2)

        loss = self.loss_function(clf_score, label)
        prob = self.softmax(clf_score)
        
        return loss, prob

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
        modeling_layer_for_clf : ModelingCLF,
        modeling_layer_for_qa : ModelingQA,
        num_labels : int,
        device
        ):
        super(Bert_QA, self).__init__()

        self.embedding_layer = embedding_layer
        self.kernel_model = kernel_model
        self.modeling_layer_clf = modeling_layer_for_clf
        self.modeling_layer_qa = modeling_layer_for_qa
      
        self.device = device
        self.hid_size = kernel_model.config.hidden_size

        self.start_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.end_clasifier = Parameter(self.get_1d_init_tensor(768))
        self.clf_clasifier = torch.nn.Linear(self.hid_size, 2)

        # self.start_clasifier = torch.nn.Linear(hid_size, hid_size)
        # self.end_clasifier = torch.nn.Linear(hid_size, hid_size)

        self.softmax = torch.nn.Softmax(dim=1)
        self.loss_function_label = torch.nn.CrossEntropyLoss()
        self.loss_function = torch.nn.CrossEntropyLoss()
        
    def forward(
        self,
        input_ids : torch.tensor,
        attention_mask : torch.tensor,
        token : torch.tensor,
        SEPind : List,
        label : torch.tensor = None, # reference dont need
        start_pos : List = None,  # reference dont need
        end_pos : List = None, # reference dont need
        return_toks : bool = False
        ) -> Tuple[torch.Tensor]:
        
        ####-----------------------------------------------------------------------------
        # this is the embedding output, the QA has answer and no answer use the same embedding layer
        # but they use the different modeling layer
        # the embedding layer is depended on which dataset you are
        # but the modeling layer is which task
        ####-----------------------------------------------------------------------------


        #### THE ENTIRE MODEL RUN AND OUTPUT --------------------------------------------
        ####-----------------------------------------------------------------------------
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        bert_output = self.kernel_model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)

        output_clf = self.modeling_layer_clf(SEPind, bert_output)  # (batch_size, 768)
        output_qa = self.modeling_layer_qa(SEPind, bert_output) # (batch_size, context_length, 768)
        ####-----------------------------------------------------------------------------
        ####-----------------------------------------------------------------------------


        #### COUNT THE PROBABILITY OF IT HAS ANSWER OR NOT ------------------------------
        ####-----------------------------------------------------------------------------
        # CLS_list = [bert_output[i,0, :] for i in range(len(SEPind))]
        # CLS_output = torch.stack(CLS_list)
        clf_score = self.clf_clasifier(output_clf) # (batch_size, 2)
        ####-----------------------------------------------------------------------------
        ####-----------------------------------------------------------------------------
 

        #### COUNT THE START AND END ----------------------------------------------------
        ####-----------------------------------------------------------------------------
        start_score = (output_qa * self.start_clasifier).sum(dim=2) # (batch_size, context_length)
        end_score = (output_qa * self.end_clasifier).sum(dim=2) # (batch_size, context_length)
        ####-----------------------------------------------------------------------------
        ####-----------------------------------------------------------------------------


        #### ONLY REFERENCE AND DON'T NEED LOSS -----------------------------------------
        ####-----------------------------------------------------------------------------
        if return_toks :
            start_tok = start_score.argmax(dim=1)
            end_tok = end_score.argmax(dim=1)

            batch_toks = []

            for i in range (len(input_ids)) :
                if clf_score[i].argmax() == 0 : # predict it has no answer
                    batch_toks.append([])
                else :
                    batch_toks.append(input_ids[i, start_tok[i]+1 : end_tok[i]+2])  # +1 +2 because of [CLS]    

            return batch_toks
        ####-----------------------------------------------------------------------------
        ####-----------------------------------------------------------------------------
        

        #### NEED LOSS ------------------------------------------------------------------
        ####-----------------------------------------------------------------------------
        loss_for_label = self.loss_function_label(clf_score, label)
        prob = self.softmax(clf_score)

        if label[0][1] == 1 :     
            # this batch data has answer, need the start and end position loss

            this_batch_size = input_ids.size(0)
            start_1hot = torch.zeros(this_batch_size, output_qa.size(1)).to(self.device)
            end_1hot = torch.zeros(this_batch_size, output_qa.size(1)).to(self.device)

            # the startpos nedd -1 because of [CLS] is get rid of during the modeling layer
            for i in range(this_batch_size):
                start_1hot[i][start_pos[i]-1] = 1 
                end_1hot[i][end_pos[i]-1] = 1 

            loss_start = self.loss_function(start_score, start_1hot)
            loss_end = self.loss_function(end_score, end_1hot)

            total_loss =  loss_for_label + (loss_start + loss_end)
        else :
            total_loss = loss_for_label

        return total_loss, prob
        ####-----------------------------------------------------------------------------
        ####-----------------------------------------------------------------------------
