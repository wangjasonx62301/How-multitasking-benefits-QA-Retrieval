import torch
from MEOW_Models.ST_model import Bert_classification, Bert_pairwise, Bert_QA
from MEOW_Models.Kernel_model import BertWithoutEmbedding
from typing import*

class MEOW_MTM:
    def __init__(
        self,
        kernel_model:BertWithoutEmbedding,
        CoLA_embedding_layer = None, CoLA_num_labels = None,
        Sentiment_embedding_layer = None, Sentiment_num_labels = None,
        MNLI_embedding_layer = None, MNLI_num_labels = None,
        RTE_embedding_layer = None, RTE_num_labels = None,
        SQuAD_embedding_layer = None,
        device = None):
                
        self.device = device
        self.kernel_model = kernel_model
        
        self.has_CoLA = CoLA_embedding_layer != None
        self.has_Sentiment = Sentiment_embedding_layer != None
        self.has_SQuAD = SQuAD_embedding_layer != None

        if self.has_CoLA :
            self.CoLA_model = Bert_classification(kernel_model, CoLA_embedding_layer, device, CoLA_num_labels)
            self.CoLA_optimizer = torch.optim.SGD(self.CoLA_model.parameters(), lr=0.0001, momentum=0.9)

        if self.has_Sentiment :
            self.Sentiment_model = Bert_classification(kernel_model, Sentiment_embedding_layer, device, Sentiment_num_labels)
            self.Sentiment_optimizer = torch.optim.SGD(self.Sentiment_model.parameters(), lr=0.0001, momentum=0.9)

        if self.has_SQuAD :
            self.SQuAD_model = Bert_QA(kernel_model, SQuAD_embedding_layer, num_labels=2, device=device)
            self.SQuAD_optimizer = torch.optim.SGD(self.SQuAD_model.parameters(), lr=0.00005, momentum=0.9)

        # self.MNLI_model = Bert_pairwise(kernel_model, MNLI_embedding_layer, device, MNLI_num_labels)
        # self.RTE_model = Bert_pairwise(kernel_model, RTE_embedding_layer, device, RTE_num_labels)
        # self.MNLI_optimizer = torch.optim.SGD(self.MNLI_model.parameters(), lr=0.0001, momentum=0.9)
        # self.RTE_optimizer = torch.optim.SGD(self.CoLA_model.parameters(), lr=0.0001, momentum=0.9)

        self.change_the_device(device)

        self.forward_dict = {'CoLA' : self.CoLA_forward,
                             'Sentiment' : self.Sentiment_forward,
                             'MNLI' : self.MNLI_forward,
                             'RTE' : self.RTE_forward,
                             'SQuAD' : self.SQuAD_forward}
        
        self.optimize_dict = {'CoLA' : self.optimize_CoLA,
                              'Sentiment' : self.optimize_Sentiment,
                              'MNLI' : self.optimize_MNLI,
                              'RTE' : self.optimize_RTE,
                              'SQuAD' : self.optimize_SQuAD}
        
    def mt_forward(self,
                   task_type : str,
                   dataset_name : str,
                   input_ids : torch.tensor, 
                   mask : torch.tensor, 
                   token_type_ids : torch.tensor, 
                   label : torch.tensor = None,
                   SEPind : List = None,
                   start_pos : List = None,
                   end_pos : List = None,
                   ):
        if(task_type == 'Classification'):
            return self.forward_dict[dataset_name](input_ids, mask, token_type_ids, label, SEPind)
        elif(task_type == 'Pairwise'):
            return self.forward_dict[dataset_name](input_ids, mask, token_type_ids, label, SEPind)
        else:
            return self.forward_dict[dataset_name](input_ids, mask, token_type_ids, label, SEPind, start_pos, end_pos)
        
    def mt_optimize(self, loss, dataset_name):
        self.optimize_dict[dataset_name](loss)

    def change_the_device(self, device):
        if self.has_CoLA :
            self.CoLA_model.to(device)
        if self.has_Sentiment :
            self.Sentiment_model.to(device)
        if self.has_SQuAD :
            self.SQuAD_model.to(device)

        # self.MNLI_model.to(device)
        # self.RTE_model.to(device)

    def optimize_CoLA(self, loss):
        optimizer = self.CoLA_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def optimize_Sentiment(self, loss):
        optimizer = self.Sentiment_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def optimize_MNLI(self, loss):
        optimizer = self.MNLI_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def optimize_RTE(self, loss):
        optimizer = self.RTE_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def optimize_SQuAD(self, loss):
        optimizer = self.SQuAD_optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def CoLA_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        return self.CoLA_model(input_ids, mask, token_type_ids, label, SEPind)
    
    def Sentiment_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        return self.Sentiment_model(input_ids, mask, token_type_ids, label, SEPind)
    
    def MNLI_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        return self.MNLI_model(input_ids, mask, token_type_ids, label, SEPind)
    
    def RTE_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        return self.RTE_model(input_ids, mask, token_type_ids, label, SEPind)
    
    def SQuAD_forward(self, input_ids, mask, token_type_ids, label, SEPind, start_pos, end_pos, return_start_end_pos = False):
        return self.SQuAD_model(input_ids, mask, token_type_ids, label, SEPind, start_pos, end_pos, return_start_end_pos)
     
    def train(self):
        if(self.has_CoLA) :
            self.CoLA_model.train()
        if(self.has_Sentiment) :
            self.Sentiment_model.train()
        if(self.has_SQuAD) :
            self.SQuAD_model.train()
        # self.MNLI_model.train()
        # self.RTE_model.train()

    def eval(self):
        if self.has_CoLA :
            self.CoLA_model.eval()
        if self.has_Sentiment :
            self.Sentiment_model.eval()
        if self.has_SQuAD :
            self.SQuAD_model.eval()

        # self.MNLI_model.eval()
        # self.RTE_model.eval()
