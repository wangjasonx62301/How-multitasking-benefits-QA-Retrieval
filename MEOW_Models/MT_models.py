import torch
from MEOW_Models.ST_model import Bert_classification, Bert_QA
from MEOW_Models.Kernel_model import BertWithoutEmbedding, ModelingQA, ModelingCLF
from MEOW_Utils.Data_utils import DataBox
from typing import*

class MEOW_MTM:
    def __init__(
        self,
        kernel_model : BertWithoutEmbedding,
        modeling_layer_for_qa : ModelingQA,
        CoLA_databox : DataBox = None,
        MNLI_databox : DataBox = None,
        SQuAD_databox : DataBox = None,
        device = None):
                
        self.device = device
        self.kernel_model = kernel_model
        
        self.has_CoLA = CoLA_databox != None
        self.has_MNLI = MNLI_databox != None
        self.has_SQuAD = SQuAD_databox != None

        #### initial all model
        #### ---------------------------------------------------------------------------------
        if self.has_CoLA :
            self.CoLA_model = Bert_classification(kernel_model,
                                                  CoLA_databox.embedding_layer, 
                                                  CoLA_databox.modeling_layer, 
                                                  CoLA_databox.label_nums, 
                                                  device)
            self.CoLA_optimizer = torch.optim.SGD(self.CoLA_model.parameters(), lr=0.00005, momentum=0.9)

        if self.has_MNLI :
            self.MNLI_model = Bert_classification(kernel_model, 
                                                  MNLI_databox.embedding_layer, 
                                                  MNLI_databox.modeling_layer, 
                                                  MNLI_databox.label_nums, 
                                                  device)
            self.MNLI_optimizer = torch.optim.SGD(self.MNLI_model.parameters(), lr=0.00005, momentum=0.9)

        if self.has_SQuAD :
            self.SQuAD_model = Bert_QA(kernel_model,
                                       SQuAD_databox.embedding_layer, 
                                       SQuAD_databox.modeling_layer,
                                       modeling_layer_for_qa,
                                       num_labels = 2, 
                                       device = device)
            self.SQuAD_optimizer = torch.optim.SGD(self.SQuAD_model.parameters(), lr=0.00005, momentum=0.9)
        #### ---------------------------------------------------------------------------------
        #### ---------------------------------------------------------------------------------

        self.change_the_device(device)

        self.forward_dict = {'CoLA' : self.CoLA_forward,
                             'MNLI' : self.MNLI_forward,
                             'SQuAD' : self.SQuAD_forward}
        
        self.optimize_dict = {'CoLA' : self.optimize_CoLA,
                              'MNLI' : self.optimize_MNLI,
                              'SQuAD' : self.optimize_SQuAD}
        
    def mt_forward(self,
                   task_type : str,
                   dataset_name : str,
                   input_ids : torch.tensor, 
                   mask : torch.tensor, 
                   token_type_ids : torch.tensor, 
                   SEPind : List,
                   label : torch.tensor = None, # if inference, don't need it
                   start_pos : List = None,  #for qa
                   end_pos : List = None,  #for qa
                   return_toks : bool = False # for qa
                   ):
        if(task_type == 'Classification'):
            return self.forward_dict[dataset_name](input_ids, mask, token_type_ids, SEPind, label)
        else:
            return self.forward_dict[dataset_name](input_ids, mask, token_type_ids, SEPind, label, start_pos, end_pos, return_toks)
        
    def mt_optimize(self, loss, dataset_name):
        self.optimize_dict[dataset_name](loss)

    def change_the_device(self, device):
        if self.has_CoLA :
            self.CoLA_model.to(device)
        if self.has_MNLI :
            self.MNLI_model.to(device)
        if self.has_SQuAD :
            self.SQuAD_model.to(device)

        # self.MNLI_model.to(device)
        # self.RTE_model.to(device)

    def optimize_CoLA(self, loss):
        optimizer = self.CoLA_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def optimize_MNLI(self, loss):
        optimizer = self.MNLI_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def optimize_SQuAD(self, loss):
        optimizer = self.SQuAD_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def CoLA_forward(self, input_ids, mask, token_type_ids, SEPind, label):
        return self.CoLA_model(input_ids, mask, token_type_ids, SEPind, label)

    def MNLI_forward(self, input_ids, mask, token_type_ids, SEPind, label):
        return self.MNLI_model(input_ids, mask, token_type_ids, SEPind, label)

    def SQuAD_forward(self, input_ids, mask, token_type_ids, SEPind, label = None, start_pos = None, end_pos = None, return_toks = False):
        return self.SQuAD_model(input_ids, mask, token_type_ids, SEPind, label, start_pos, end_pos, return_toks)
     
    def train(self):
        if(self.has_CoLA) :
            self.CoLA_model.train()
        if(self.has_MNLI) :
            self.MNLI_model.train()
        if(self.has_SQuAD) :
            self.SQuAD_model.train()
        # self.MNLI_model.train()
        # self.RTE_model.train()

    def eval(self):
        if self.has_CoLA :
            self.CoLA_model.eval()
        if self.has_MNLI :
            self.MNLI_model.eval()
        if self.has_SQuAD :
            self.SQuAD_model.eval()

        # self.MNLI_model.eval()
        # self.RTE_model.eval()
