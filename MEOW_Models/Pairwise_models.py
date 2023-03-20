import torch
from transformers import BertModel
from MEOW_Models.MT_models import*
from MEOW_Utils.QA_utils import*

class Bert_pairwise(torch.nn.Module):
    def __init__(self, model:BertWithoutEmbedding, embedding_layer, device, num_labels):
        super(Bert_pairwise, self).__init__()
        
        self.device = device
        self.model = model
        self.embedding_layer = embedding_layer
        self.dropout = torch.nn.Dropout(p=0.1)
        self.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
        
        self.loss_function = torch.nn.CrossEntropyLoss() #會自己 softmax
        self.softmax = torch.nn.Softmax(dim=1)
        
        
    # return 出(batch, seq_len)
    def forward(self, input_ids, attention_mask, token, label, SEPind):
        embedding_output = self.embedding_layer(input_ids=input_ids, token_type_ids=token)
        outputs = self.model(embedding_output=embedding_output, input_ids=input_ids, attention_mask=attention_mask, token_type_ids = token)
    
        # last hidden layer
        last_hidden_layer = outputs.last_hidden_state  # (batch_size, sequence_length, hidden_size:768)

        # sep_list = [last_hidden_layer[i,SEPind[i], :] for i in range(len(SEPind))]
        sep_list = [last_hidden_layer[i,0, :] for i in range(len(SEPind))]
        for_sep = torch.stack(sep_list)

        sep_output = self.dropout(for_sep)
        logits = self.classifier(sep_output)

        loss = self.loss_function(logits, label)
        prob = self.softmax(logits)
        
        return loss, prob
    
    