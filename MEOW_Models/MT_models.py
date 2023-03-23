import torch
from MEOW_Models.ST_model import*

class MEOW_MTM:
    def __init__(
        self,
        kernel_model:BertWithoutEmbedding,
        CoLA_embedding_layer, CoLA_num_labels,
        Sentiment_embedding_layer, Sentiment_num_labels,
        MNLI_embedding_layer, MNLI_num_labels,
        RTE_embedding_layer, RTE_num_labels,
        SQuAD_embedding_layer,
        device):
                
        self.device = device
        self.kernel_model = kernel_model

        # self.CoLA_embedding_layer = CoLA_embedding_layer
        # self.Sentiment_embedding_layer = Sentiment_embedding_layer
        # self.MNLI_embedding_layer = MNLI_embedding_layer
        # self.RTE_embedding_layer = RTE_embedding_layer
        # self.SQuAD_embedding_layer = SQuAD_embedding_layer

        self.CoLA_model = Bert_classification(kernel_model, CoLA_embedding_layer, device, CoLA_num_labels)
        self.Sentiment_model = Bert_classification(kernel_model, Sentiment_embedding_layer, device, Sentiment_num_labels)
        self.MNLI_model = Bert_pairwise(kernel_model, MNLI_embedding_layer, device, MNLI_num_labels)
        self.RTE_model = Bert_pairwise(kernel_model, RTE_embedding_layer, device, RTE_num_labels)
        self.SQuAD_model = Bert_QA(kernel_model, SQuAD_embedding_layer, device)

        self.CoLA_optimizer = torch.optim.SGD(self.CoLA_model.parameters(), lr=0.0001, momentum=0.9)
        self.Sentiment_optimizer = torch.optim.SGD(self.Sentiment_model.parameters(), lr=0.0001, momentum=0.9)
        self.MNLI_optimizer = torch.optim.SGD(self.MNLI_model.parameters(), lr=0.0001, momentum=0.9)
        self.RTE_optimizer = torch.optim.SGD(self.CoLA_model.parameters(), lr=0.0001, momentum=0.9)
        self.SQuAD_optimizer = torch.optim.SGD(self.SQuAD_model.parameters(), lr=0.0001, momentum=0.9)

        self.change_the_device()

    def change_the_device(self):
        device = self.device
        self.CoLA_model.to(device)
        self.Sentiment_model.to(device)
        self.MNLI_model.to(device)
        self.RTE_model.to(device)
        self.SQuAD_model.to(device)

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
    
    def CoLA_forward(self, input_ids, mask, token_type_ids, label):
        loss, prob = self.CoLA_model(input_ids, mask, token_type_ids, label=label)
        return loss, prob
    
    def Sentiment_forward(self, input_ids, mask, token_type_ids, label):
        loss, prob = self.Sentiment_model(input_ids, mask, token_type_ids, label=label)
        return loss, prob
    
    def MNLI_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        loss, prob = self.MNLI_model(input_ids, mask, token_type_ids, label, SEPind)
        return loss, prob
    
    def RTE_forward(self, input_ids, mask, token_type_ids, label, SEPind):
        loss, prob = self.RTE_model(input_ids, mask, token_type_ids, label, SEPind)
        return loss, prob
    
    def SQuAD_forward(self, input_ids, mask, token_type_ids, SEPind, start_pos, end_pos):
        loss = self.SQuAD_model(input_ids, mask, token_type_ids, SEPind, start_pos, end_pos)
        return loss
    
    def train(self):
        self.CoLA_model.train()
        self.Sentiment_model.train()
        self.MNLI_model.train()
        self.RTE_model.train()
        self.SQuAD_model.train()
