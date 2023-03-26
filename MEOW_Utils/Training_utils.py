import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from MEOW_Models.MT_models import MEOW_MTM

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

def QA_training(MEOW_model : MEOW_MTM, 
                iter,
                device,
                dataset_name,
                do_optimize = False
                ):
        input_ids, mask, token, SEPind, Start_pos, End_pos = next(iter)

        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        
        loss = MEOW_model.mt_forward(task_type= 'QA', 
                                     dataset_name = dataset_name,
                                     input_ids = input_ids, 
                                     mask = mask, 
                                     token_type_ids = token,
                                     SEPind = SEPind, 
                                     start_pos = Start_pos,
                                     end_pos = End_pos)
        
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)
        
        return loss

def Pairwise_training(MEOW_model : MEOW_MTM, 
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

def Classifiaction_training(MEOW_model : MEOW_MTM, 
                            iter,
                            device, 
                            dataset_name,
                            do_optimize = False
                            ):
        input_ids, mask, token, label = next(iter)
        
        input_ids = input_ids.to(device)
        mask = mask.to(device)
        token = token.to(device)
        label = label.to(device)
        
        loss, prob = MEOW_model.mt_forward(task_type = 'Classification',
                                           dataset_name = dataset_name,
                                           input_ids = input_ids,
                                           mask = mask,
                                           token_type_ids = token,
                                           label = label)
        
        acur = count_correct_num(prob, label)
        if do_optimize:
            MEOW_model.mt_optimize(loss=loss, dataset_name=dataset_name)
        
        return loss, prob, acur

training_dict = {'CoLA' : Classifiaction_training,
                 'Sentiment' : Classifiaction_training,
                 'MNLI' : Pairwise_training,
                 'RTE' : Pairwise_training,
                 'SQuAD' : QA_training }

def Training(MEOW_model : MEOW_MTM, 
             iter,
             device, 
             dataset_name,
             do_optimize = False
             ):
    return training_dict[dataset_name](MEOW_model, iter, device, dataset_name, do_optimize)
