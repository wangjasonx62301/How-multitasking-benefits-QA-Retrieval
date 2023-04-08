import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from MEOW_Models.MT_models import MEOW_MTM
from pandas import DataFrame
from transformers import BertTokenizer
from MEOW_Utils.Data_utils import count_the_TKbeg_and_TKend
from evaluate import load

def plot_diagram(H, epoch_num, has_accuracy=False):
    # tensor to float
    Train_loss = [float(i) for i in H['train_loss']]
    
    if has_accuracy:
        Train_acur = [float(i) for i in H['train_acc']]
    Test_loss = [float(i) for i in H['test_loss']]
    Test_acur = [float(i) for i in H['test_acc']]

    # loss
    plt.figure()
    plt.title("Loss")
    plt.xlabel("EPOCH")
    plt.ylabel("Loss")
    plt.plot(Train_loss, label="test_loss")
    plt.plot(Test_loss, label="test_loss")
    plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
    plt.show()
    
    if has_accuracy:
        # accuracy
        plt.figure()
        plt.title("Test Accuracy")
        plt.xlabel("EPOCH")
        plt.ylabel("Accuracy")
        plt.plot(Train_acur, label="test_acc")
        plt.plot(Test_acur, label="test_acc")
        plt.xticks(np.arange(epoch_num), range(1,epoch_num+1,1))
        plt.show()

def count_correct_num(prob : torch.tensor, label : torch.tensor):
    predict = torch.argmax(prob, dim=1)
    label = torch.argmax(label, dim=1)
    correct_num = (predict == label).type(torch.int).sum()
    return correct_num

def QA_running(MEOW_model : MEOW_MTM, 
                iter,
                device,
                dataset_name,
                do_optimize = False
                ):
        input_ids, mask, token, label, SEPind, Start_pos, End_pos = next(iter)

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type= 'QA', 
                                            dataset_name = dataset_name,
                                            input_ids = input_ids, 
                                            mask = mask, 
                                            token_type_ids = token,
                                            label=label,
                                            SEPind = SEPind, 
                                            start_pos = Start_pos,
                                            end_pos = End_pos)
        
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)
        
        return loss, prob

def Pairwise_running(MEOW_model : MEOW_MTM, 
                      iter,
                      device,
                      dataset_name,
                      do_optimize = False
                      ):
        input_ids, mask, token, label, SEPind = next(iter)

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type='Pairwise',
                                           dataset_name = dataset_name,
                                           input_ids = input_ids,
                                           mask = mask,
                                           token_type_ids = token, 
                                           label = label,
                                           SEPind = SEPind)
        
        acur = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)

        return loss, prob, acur

def Classifiaction_running(MEOW_model : MEOW_MTM, 
                            iter,
                            device, 
                            dataset_name,
                            do_optimize = False
                            ):
        input_ids, mask, token, label, SEPind = next(iter)
        
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type = 'Classification',
                                           dataset_name = dataset_name,
                                           input_ids = input_ids,
                                           mask = mask,
                                           token_type_ids = token,
                                           label = label,
                                           SEPind = SEPind
                                           )
        
        acur = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)
        
        return loss, prob, acur

running_dict = {'CoLA' : Classifiaction_running,
                 'Sentiment' : Classifiaction_running,
                 'MNLI' : Pairwise_running,
                 'RTE' : Pairwise_running,
                 'SQuAD' : QA_running }

def Training(MEOW_model : MEOW_MTM, 
             iter,
             device, 
             dataset_name,
             ):
    return running_dict[dataset_name](MEOW_model, iter, device, dataset_name, do_optimize = True)

def Test(MEOW_model : MEOW_MTM, 
             iter,
             device, 
             dataset_name,
             ):
    return running_dict[dataset_name](MEOW_model, iter, device, dataset_name, do_optimize = False)

def count_F1_score(MEOW_model : MEOW_MTM, 
                   df : DataFrame, 
                   tokenizer : BertTokenizer,
                   device):
    
    squad_metric = load("squad_v2")

    prediction = []
    reference = []

    for i in range(len(df)):
        EC = tokenizer.encode_plus(df['context'][i], df['question'][i])

        SEPind = [len(tokenizer.tokenize(df['context'][i])) + 1]
        
        input_ids = torch.tensor([EC['input_ids']])  # 要讓他升一個維度 表示batch
        mask = torch.tensor([EC['attention_mask']])
        token = torch.tensor([EC['token_type_ids']])
        label = [[0.] * 2]
        label[0][df['label'][i]] = 1.

        label = torch.tensor(label, dtype=torch.float)

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)

        Start_pos, End_pos = count_the_TKbeg_and_TKend(df['context'][i], df['answer_start'][i], df['text'][i], tokenizer)
        
        Start_pos = [Start_pos]
        End_pos = [End_pos]

        start, end = MEOW_model.SQuAD_forward(input_ids=input_ids, 
                                            mask=mask, 
                                            token_type_ids=token,
                                            label=label,
                                            SEPind=SEPind,
                                            start_pos=Start_pos, 
                                            end_pos=End_pos,
                                            return_start_end_pos=True)
        
        context = tokenizer.convert_ids_to_tokens(input_ids[0])
        str_pred = tokenizer.convert_tokens_to_string(context[start[0]:end[0]+1])

        NA_prob = 0.
        if(str_pred == ""):
            NA_prob = 1.

        # squad_v2_metric = load("squad_v2")
        # predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.}]
        # references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]

        print(str_pred)
        print(df['text'][i])

        prediction.append({'prediction_text': str_pred, 'id': df['index'][i], 'no_answer_probability': NA_prob})
        reference.append({'id': df['index'][i], 'answers': {'answer_start': [df['answer_start'][i]], 'text': df['text'][i]}})

    
    results = squad_metric.compute(predictions=prediction, references=reference)
    return results['f1']
