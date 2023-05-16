import torch
from MEOW_Models.ST_model import Bert_classification, Bert_QA
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
from MEOW_Utils.Data_utils import DataBox
from typing import*

class MEOW_MTM(torch.nn.Module):
    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        modeling_layer_for_qa : ModelingQA,
        qa_databox : DataBox,
        support_databox_list : torch.nn.ModuleList = None,
        device = None):

        super(MEOW_MTM, self).__init__()
                
        self.device = device
        self.kernel_model = kernel_model
        self.support_data_num = len(support_databox_list)

        self.support_key = torch.nn.Linear(768, 768)
        self.target_query = torch.nn.Linear(768, 768) ## use the output of SQuAD's output

        #### initial all model
        #### ---------------------------------------------------------------------------------
        self.support_modulelist = torch.nn.ModuleList()
        self.optimizer_list = []     # optimizer_list[-1] is QA optimizer

        #### for support data ----------------------
        for i in range(self.support_data_num):
            self.support_modulelist.append( Bert_classification(kernel_model, 
                                                                 support_databox_list[i].embedding_layer,
                                                                 support_databox_list[i].clf_modeling_layer,
                                                                 support_databox_list[i].label_nums,
                                                                 device) )
            self.optimizer_list.append(torch.optim.SGD(self.support_modulelist[0].parameters(), lr=0.00005, momentum=0.9))
        
        
        #### ---------------------------------------
        #### ---------------------------------------
         
        #### for target data -----------------------
        self.SQuAD_model = Bert_QA(kernel_model,
                                    qa_databox.embedding_layer, 
                                    qa_databox.clf_modeling_layer,
                                    modeling_layer_for_qa,
                                    num_labels = 2,
                                    support_modulelist = self.support_modulelist,
                                    support_key = self.support_key,
                                    target_query = self.target_query,
                                    device = device)
        self.SQuAD_optimizer = torch.optim.SGD(self.SQuAD_model.parameters(), lr=0.00005, momentum=0.9)
        #### ---------------------------------------
        #### ---------------------------------------

    def mt_forward(self,
                   dataset_ind : int,
                   input_ids : torch.tensor, 
                   mask : torch.tensor, 
                   token_type_ids : torch.tensor, 
                   SEPind : List,
                   label : torch.tensor = None, # if inference, don't need it
                   start_pos : List = None,  #for qa
                   end_pos : List = None,  #for qa
                   return_toks : bool = False # for qa
                   ):
        
        if(dataset_ind < self.support_data_num): ## is clf task
            return self.support_modulelist[dataset_ind](input_ids, mask, token_type_ids, SEPind, label)
        return self.SQuAD_model(input_ids, mask, token_type_ids, SEPind, label, start_pos, end_pos, return_toks)

    def mt_optimize(self, loss, dataset_ind):
        if dataset_ind == self.support_data_num :
            optimizer = self.SQuAD_optimizer
        else :
            optimizer = self.optimizer_list[dataset_ind]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
