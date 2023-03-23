import copy
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import json
from MEOW_Utils import QA_utils, Pairwise_utils, Classification_utils
from torch.utils.data import DataLoader

# other helper function
def get_balanced_df(df, column_name):
    value_count = df.value_counts(column_name)
    key_arr = value_count.keys()
    balanced_df = pd.DataFrame()

    min_num = 99999999

    for key in key_arr:
        min_num = min(min_num, value_count[key])

    for key in key_arr:
        tmp_df = df[df[column_name] == key]
        tmp_df = tmp_df.sample(n=min_num, random_state=42)
        balanced_df = pd.concat([balanced_df, tmp_df])

    balanced_df = shuffle(balanced_df)
    balanced_df = balanced_df.reset_index(drop=True)
    return balanced_df

def squad_json_to_dataframe_train(input_file_path, record_path = ['data','paragraphs','qas','answers'],
                           verbose = 1):
    if verbose:
        print("Reading the json file")    
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    js = pd.io.json.json_normalize(file , record_path )
    m = pd.io.json.json_normalize(file, record_path[:-1])
    r = pd.io.json.json_normalize(file,record_path[:-2])
    
    #combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx  = np.repeat(m['id'].values,m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([ m[['id','question','context']].set_index('id'),js.set_index('q_idx')],1,sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

def count_the_TKbeg_and_TKend(context, ans_start, answer, tokenizer):
    # the parameters is context, answer_start, text in the dataframe
    # split the context from answer start, and get the token start
    if(ans_start == -1):
        return 0, 0
    
    s1 = context[0:ans_start]

    TKstart = len(tokenizer.tokenize(s1))
    TKanslen = len(tokenizer.tokenize(answer))
    TKend = TKstart + TKanslen - 1

    return TKstart+1, TKend+1 # +1 because of [CLS]

#get dataframe
def get_MNLI_df(file_path, tokenizer, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]

    balanced_df = get_balanced_df(train_df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    train_df = balanced_df
    
    if 'Unnamed: 0' in train_df.keys():
        train_df = train_df.drop('Unnamed: 0', axis=1)

    train_df['label_name'] = train_df['label']
    train_df['label'] = train_df['label_name'].replace({'neutral':1, 'entailment':0, 'contradiction':2})

    train_df['SEP_ind'] = train_df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    train_df['context2'] = train_df['context2'].apply(lambda x : x if type(x) is str else '')

    return train_df

def get_RTE_df(file_path, tokenizer, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]

    balanced_df = get_balanced_df(train_df, column_name='label')
    balanced_df = balanced_df.reset_index(drop=True)
    train_df = balanced_df
    
    if 'Unnamed: 0' in train_df.keys():
        train_df = train_df.drop('Unnamed: 0', axis=1)

    train_df['label_name'] = train_df['label']
    train_df['label'] = train_df['label_name'].replace({'not_entailment':0, 'entailment':1})

    train_df['SEP_ind'] = train_df['context1'].apply(lambda x : len(tokenizer.tokenize(x))+1) # +1 is [CLS]
    train_df['context2'] = train_df['context2'].apply(lambda x : x if type(x) is str else '')

    return train_df

def get_CoLA_df(file_path, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]
    #print(train_df)
    balanced_df = get_balanced_df(train_df, 'label')
    if 'Unnamed: 0' in train_df.keys():
        balanced_df = balanced_df.drop('Unnamed: 0', axis=1)
    #print(balanced_df)

    train_df = balanced_df
    return train_df

def get_Sentiment_df(file_path, data_size = 0):
    train_df = pd.read_csv(file_path)
    if(data_size != 0):
        train_df = train_df[0:data_size]

    balanced_df = get_balanced_df(train_df, 'label')
    if 'Unnamed: 0' in train_df.keys():
        balanced_df = balanced_df.drop('Unnamed: 0', axis=1)
    train_df = balanced_df

    train_df['label_name'] = train_df['label']
    train_df['label'] = train_df['label_name'].replace({'Extremely Positive':0, 'Positive':1, 'Neutral':2, 'Negative':3, 'Extremely Negative':4})

    return train_df

def get_SQuAD_df(file_path, tokenizer, data_size = 0):
    #處理好 dataframe
    df_train = pd.read_csv(file_path)
    if(data_size != 0):
        df_train = df_train[0:data_size]

    # question,context

    for i in range(len(df_train)):
        if( len(tokenizer.tokenize(df_train['question'][i])) + len(tokenizer.tokenize(df_train['context'][i])) >= 510 ):
            df_train = df_train.drop(i, axis=0)
            i = i-1
    
    df_train = df_train.reset_index(drop=True)
    
    df_train['answer_start'] = df_train['answer_start'].apply(lambda x : -1 if np.isnan(x) else int(x))
    df_train['text'] = df_train['text'].apply(lambda x : x if type(x) is str else '')

    df_train['TKstart'] = pd.Series([0] * len(df_train))
    df_train['TKend'] = pd.Series([0] * len(df_train))

    for i in range(len(df_train)):
        df_train['TKstart'][i], df_train['TKend'][i] = count_the_TKbeg_and_TKend(df_train.iloc[i]['context'], df_train.iloc[i]['answer_start'], df_train.iloc[i]['text'], tokenizer)

    if ('index' in df_train.keys()):
        df_train = df_train.drop('index', axis=1)
    if ('c_id' in df_train.keys()):
        df_train = df_train.drop('c_id', axis=1)
    if 'Unnamed: 0' in df_train.keys():
        df_train = df_train.drop('Unnamed: 0', axis=1)

    df_train['SEP_ind'] = df_train['context'].apply(lambda x : len(tokenizer.tokenize(x))+1)
    
    return df_train

#get dataset
def get_MNLI_dataset(df_MNLI, tokenizer, num_labels):
    return Pairwise_utils.Pairwise_dataset(df_MNLI, tokenizer, num_labels)

def get_RTE_dataset(df_RTE, tokenizer, num_labels):
    return Pairwise_utils.Pairwise_dataset(df_RTE, tokenizer, num_labels)

def get_CoLA_dataset(df_CoLA, tokenizer, num_labels):
    return Classification_utils.Classification_dataset(df_CoLA, tokenizer, num_labels)

def get_Sentiment_dataset(df_Sentiment, tokenizer, num_labels):
    return Classification_utils.Classification_dataset(df_Sentiment, tokenizer, num_labels)

def get_SQuAD_dataset(df_SQuAD, tokenizer):
    return QA_utils.QAdataset(df_SQuAD, tokenizer=tokenizer)

#get dataloader
def get_MNLI_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Pairwise_utils.collate_batch)

def get_RTE_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Pairwise_utils.collate_batch)

def get_CoLA_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_utils.collate_batch)

def get_Sentiment_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=Classification_utils.collate_batch)

def get_SQuAD_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=QA_utils.collate_batch)
