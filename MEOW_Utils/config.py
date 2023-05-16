import torch

DATA_NAME = {0:'CoLA', 1:'MNLI', 2:'QNLI', 3:'SQuAD'}
DATA_IND = {'ClLA':0, 'MNLI':1, 'QNLI':2, 'SQuAD':3}

Target_BATCH_SIZE = 8
Support_BATCH_SIZE = 6

ORG_FILE_PATH_CoLA = r'Dataset_infile\CoLA_Prompt.csv'
ORG_FILE_PATH_MNLI = r'Dataset_infile\MNLI.csv'
ORG_FILE_PATH_QNLI = r'Dataset_infile\QNLI.csv' 
ORG_FILE_PATH_SQuAD = r'Dataset_infile\SQuAD.csv'

INPUT_FILE_PATH_CoLA = r'Dataset_infile\_CoLA.csv'
INPUT_FILE_PATH_MNLI = r'Dataset_infile\_MNLI.csv'
INPUT_FILE_PATH_QNLI = r'Dataset_infile\_QNLI.csv'
INPUT_FILE_PATH_SQuAD = r'Dataset_infile\_SQuAD.csv'

# INPUT_FILE_PATH_CoLA = r'/kaggle/input/qwertttt/Dataset_infile/_CoLA.csv'
# INPUT_FILE_PATH_MNLI = r'/kaggle/input/qwertttt/Dataset_infile/_MNLI.csv'
# INPUT_FILE_PATH_SQuAD = r'/kaggle/input/qwertttt/Dataset_infile/_SQuAD.csv'
# INPUT_FILE_PATH_QNLI = r'/kaggle/input/qwertttt/Dataset_infile/_QNLI.csv'

EPOCH_NUM = 14
SUPPORT_DATA_NUM = 3

SQuAD_DATASIZE = 100000
CoLA_DATASIZE = 8550  # max 8550
MNLI_DATASIZE = 15000  # max 15000
QNLI_DATASIZE = 15000 

TEST_SIZE = 0.3
PRETRAINED_MODULE_NAME = 'bert-base-uncased'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
