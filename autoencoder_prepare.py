import pandas as pd
import torch.utils.data
import matplotlib.pyplot as plt
"""import resource"""
import sys
sys.path.insert(1, 'src')
import pandas as pd
from src.data import *
import numpy as np
import pickle
import pandas as pd
import numpy as np
import torch
import pickle
from torch import nn
import time
import sys
sys.path.insert(1, 'src')
from src.model import Model
from src.utils import *
from src.data import *
from train_baseline import *

model_name = 's1.model'
id_bank_file_name = 's1.pickle'


parser = create_arg_parser()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20

my_id_bank_s1 = Central_ID_Bank()

# load gmf model which includes user and bugroupname embeddings
# load pretrained model
model_dir = f'checkpoints/{model_name}'
id_bank_dir = f'checkpoints/{id_bank_file_name}'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank_s1 = pickle.load(centralid_file)

gmf_model_s1 = Model(args, my_id_bank_s1)
gmf_model_s1.load(model_dir)

#LOAD S2

model_name = 's2.model'
id_bank_file_name = 's2.pickle'


parser = create_arg_parser()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20
print(args)
my_id_bank_s2 = Central_ID_Bank()

# load gmf model which includes user and bugroupname embeddings
# load pretrained model
model_dir = f'checkpoints/{model_name}'
id_bank_dir = f'checkpoints/{id_bank_file_name}'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank_s2 = pickle.load(centralid_file)

gmf_model_s2 = Model(args, my_id_bank_s2)
gmf_model_s2.load(model_dir)


#LOAD S3

model_name = 's3.model'
id_bank_file_name = 's3.pickle'

parser = create_arg_parser()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20

my_id_bank_s3 = Central_ID_Bank()

# load gmf model which includes user and bugroupname embeddings
# load pretrained model
model_dir = f'checkpoints/{model_name}'
id_bank_dir = f'checkpoints/{id_bank_file_name}'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank_s3 = pickle.load(centralid_file)

gmf_model_s3 = Model(args, my_id_bank_s3)
gmf_model_s3.load(model_dir)



#t1 load

model_name = 't1.model'
id_bank_file_name = 't1.pickle'

parser = create_arg_parser()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20

my_id_bank_t1 = Central_ID_Bank()

# load gmf model which includes user and bugroupname embeddings
# load pretrained model
model_dir = f'checkpoints/{model_name}'
id_bank_dir = f'checkpoints/{id_bank_file_name}'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank_t1 = pickle.load(centralid_file)

gmf_model_t1 = Model(args, my_id_bank_t1)
gmf_model_t1.load(model_dir)


#t2 load

model_name = 't2.model'
id_bank_file_name = 't2.pickle'


parser = create_arg_parser()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20

my_id_bank_t2 = Central_ID_Bank()

# load gmf model which includes user and bugroupname embeddings
# load pretrained model
model_dir = f'checkpoints/{model_name}'
id_bank_dir = f'checkpoints/{id_bank_file_name}'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank_t2 = pickle.load(centralid_file)

gmf_model_t2 = Model(args, my_id_bank_t2)
gmf_model_t2.load(model_dir)



s1_train5 = pd.read_csv("DATA/s1/train_5core.tsv", sep="\t")
s2_train5 = pd.read_csv("DATA/s2/train.tsv", sep="\t")
s3_train5 = pd.read_csv("DATA/s3/train.tsv", sep="\t")
t1_train5 = pd.read_csv("DATA/t1/train.tsv", sep="\t")
t2_train5 = pd.read_csv("DATA/t2/train_5core.tsv", sep="\t")
t1_valid = pd.read_csv("DATA/t1/valid_run.tsv", sep="\t")

id_bank_dir = f'./t1_s1-s2-s3_s1-s2-s3-t2_train_all.pickle'

with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank = pickle.load(centralid_file)

users, items = [], []
with open("DATA/t1/valid_run.tsv", 'r') as f:
    for line in f:
        linetoks = line.split('\t')
        user_id = linetoks[0]
        item_ids = linetoks[1].strip().split(',')
        for cindex, item_id in enumerate(item_ids):
            users.append(user_id)
            items.append(item_id)
print(len(items))

all_train = pd.concat([s1_train5,s2_train5,s3_train5,t1_train5,t2_train5])
print(len(list(all_train.itemId.unique())))

markets = ["s1","s2","s3","t1","t2"]

unique_market_item_emd = []

for market in markets:
  item_list = eval(f'list({market}_train5.itemId.unique())')
  for item_id in item_list:
    item_index = eval(f'my_id_bank_{market}.query_item_index(item_id)')
    item_embedding = eval(f'gmf_model_{market}.model.embedding_item.weight[item_index].detach().cpu().numpy()')
    unique_market_item_emd.append([market,item_id,item_embedding])

unique_market_item_df = pd.DataFrame(unique_market_item_emd, columns =['market', 'item_id', 'embedding_vector'])

unique_market_item_df.head()

unique_market_item_df.to_csv("unique_market_item.csv")

pair_markets = [("s1","s2"),
                ("s1","s3"),
                ("s2","s3"),
                ("s1","t1"),
                ("s1","t2"),
                ("s2","t1"),
                ("s2","t2"),
                ("s3","t1"),
                ("s3","t2"),
                ("t1","t2")]

result = []

# iterate markets
for market_names in pair_markets:

  # find item intersections between two markets
  print(market_names[0] , market_names[1])
  first_item_list = eval(f'list({market_names[0]}_train5.itemId.unique())')
  second_item_list = eval(f'list({market_names[1]}_train5.itemId.unique())')
  intersection = [e for e in  first_item_list if e in second_item_list]
  print(len(intersection))

  for item_id in intersection:
    item_index_first = eval(f'my_id_bank_{market_names[0]}.query_item_index( str(item_id) )')
    item_index_second = eval(f'my_id_bank_{market_names[1]}.query_item_index(item_id)')
    item_embedding_first = eval(f'gmf_model_{market_names[0]}.model.embedding_item.weight[item_index_first].detach().cpu().numpy()')
    item_embedding_second = eval(f'gmf_model_{market_names[1]}.model.embedding_item.weight[item_index_second].detach().cpu().numpy()')

    #print(item_embedding_first)
    #print(item_embedding_second)

    result.append([item_id, market_names[0], market_names[1], item_embedding_first, item_embedding_second])

df = pd.DataFrame(result, columns =['item_id', 'market_1', 'market_2', 'embedding_vector_1', 'embedding_vector_2'])

print(df.head())

df = df.sample(frac=1).reset_index(drop=True)

print(df.head())

print(len(df))

df.to_csv("pair_embeddings.csv")

print(len(s1_train5.itemId.unique()))

print(len(s2_train5.itemId.unique()))
print(len(s3_train5.itemId.unique()))

print(len(s1_train5.userId.unique()))
print(len(s1_train5.itemId.unique()))
print(gmf_model_s1.model.embedding_user)
print(gmf_model_s1.model.embedding_item)