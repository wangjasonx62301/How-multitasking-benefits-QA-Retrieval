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
import collections 

def plot_diagram(H, epoch_num, has_accuracy=False):
    # tensor to float
    Train_loss = [float(i) for i in H['train_loss']]
    Test_loss = [float(i) for i in H['test_loss']]

    if has_accuracy:
        Train_acur = [float(i) for i in H['train_acc']]
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
                do_optimize = False,
                return_toks = False
                ):
        input_ids, mask, token, label, SEPind, Start_pos, End_pos = next(iter)

        orgdevice = input_ids.device

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type= 'QA', 
                                            dataset_name = dataset_name,
                                            input_ids = input_ids, 
                                            mask = mask, 
                                            token_type_ids = token,
                                            SEPind = SEPind,
                                            label=label,
                                            start_pos = Start_pos,
                                            end_pos = End_pos,
                                            return_toks = return_toks)
        
        correct_num = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)

        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)
        
        return loss, prob, correct_num

def Classifiaction_running(MEOW_model : MEOW_MTM, 
                            iter,
                            device, 
                            dataset_name,
                            do_optimize = False
                            ):
        input_ids, mask, token, label, SEPind = next(iter)

        orgdevice = input_ids.device
        
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type = 'Classification',
                                           dataset_name = dataset_name,
                                           input_ids = input_ids,
                                           mask = mask,
                                           token_type_ids = token,
                                           SEPind = SEPind,
                                           label = label
                                           )
        
        correct = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)


        # to provent the cuda from out of memory
        # use to orgdevice to releace the memory allocated to tensor
        input_ids = input_ids.to(orgdevice)
        mask = mask.to(orgdevice)
        token = token.to(orgdevice)
        label = label.to(orgdevice)
        
        return loss, prob, correct

def count_F1_score(MEOW_model : MEOW_MTM, 
                   df : DataFrame, 
                   tokenizer : BertTokenizer,
                   device):
    
    score = 0

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

        toks = MEOW_model.SQuAD_forward(input_ids=input_ids, 
                                            mask=mask, 
                                            token_type_ids=token,
                                            SEPind=SEPind,
                                            label=label,
                                            start_pos=Start_pos, 
                                            end_pos=End_pos,
                                            return_toks=True)
    

        ans_toks = tokenizer.tokenize(df['text'][i])
        print(ans_toks)

        pred_toks = tokenizer.convert_ids_to_tokens(toks[0])
        print(pred_toks)
        print('')

        score += compute_f1(ans_toks, pred_toks)

    print(score / len(df))
    print('')
    return 0

def compute_f1(targ_toks : list, pred_toks : list):    
    common = collections.Counter(targ_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(targ_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(targ_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(targ_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

