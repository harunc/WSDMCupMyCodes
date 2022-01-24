import torch.utils.data
import matplotlib.pyplot as plt
"""import resource"""
import sys
sys.path.insert(1, 'src')
import pandas as pd
from src.data import *
import numpy as np
import pickle
from src.model import Model
from src.utils import *
from src.data import *
from train_baseline import *


class AE(torch.nn.Module):
    def __init__(self):
      super().__init__()

      # Building an linear encoder with Linear
      # layer followed by Relu activation function
      self.encoder = torch.nn.Sequential(
        torch.nn.Linear(20, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 60),
        torch.nn.ReLU(),
        # torch.nn.Linear(120, 60),
        # torch.nn.ReLU(),
        torch.nn.Linear(60, 20),
      )

      # Building an linear decoder with Linear
      # layer followed by Relu activation function

      self.decoder = torch.nn.Sequential(
        torch.nn.Linear(20, 60),
        torch.nn.ReLU(),
        torch.nn.Linear(60, 40),
        torch.nn.ReLU(),
        #torch.nn.Linear(120, 240),
        #torch.nn.ReLU(),
        torch.nn.Linear(40, 20)
      )

    def forward(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

    # Model Initialization
model = AE()

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             weight_decay=1e-9)
epochs = 1
losses = [0.0]
c = 0
labels = []
outputs = []



input_embeddings = pd.read_csv("pair_embeddings.csv")

input_embeddings["embedding_vector_1"] = input_embeddings["embedding_vector_1"].apply(
    lambda x: [float(x) for x in x.replace("[", "").replace ("]", "").split()])
input_embeddings["embedding_vector_2"] = input_embeddings["embedding_vector_2"].apply(
    lambda x: [float(x) for x in x.replace("[", "").replace ("]", "").split()])

input_embeddings.head()


for epoch in range(epochs):

    print("epoch: ", epoch)
    total_loss = 0.0

    for i, row in input_embeddings.iterrows():
        item_in_source = row['embedding_vector_1']
        item_in_target = row['embedding_vector_2']
        item_in_source = torch.tensor(item_in_source)
        item_in_target = torch.tensor(item_in_target)

        reconstructed = model(item_in_source)
        loss = loss_function(reconstructed, item_in_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss = loss + total_loss

        # labels.append(item_in_target)
        # outputs.append(reconstructed)

    if epoch > 0 and epoch % 5 == 0:
        model_name = "./ae_models/ae_" + str(epoch) + ".model"
        torch.save(model.state_dict(), model_name)

    losses.append(total_loss)
    print(total_loss)

model_ae = AE()

state_dict = torch.load("./ae_models/ae_20.model")
model_ae.load_state_dict(state_dict, strict=False)

model_ae.eval()

eval_input_embeddings = pd.read_csv("unique_market_item.csv")

eval_input_embeddings["embedding_vector"] = eval_input_embeddings["embedding_vector"].apply(
    lambda x : [float(x) for x in x.replace("[", "").replace ("]", "").split()])

bottleneck_dict = {}

for i, row in eval_input_embeddings.iterrows():

    item_id = row['item_id']

    item_emb = row['embedding_vector']
    item_emb = torch.tensor(item_emb)
    bottleneck = model_ae.encoder(item_emb)
    bottleneck = bottleneck.tolist()

    if item_id not in bottleneck_dict:
        init_emb_list = [bottleneck]
        bottleneck_dict[item_id] = init_emb_list
    else:
        temp = bottleneck_dict[item_id]
        temp.append(bottleneck)
        bottleneck_dict[item_id] = temp

bottleneck_dict.keys()

bottleneck_dict_end = { key: list(np.mean(bottleneck_dict[key], axis=0)) for key in bottleneck_dict}

print((bottleneck_dict['P1000023'][0][1] + bottleneck_dict['P1000023'][1][1] + bottleneck_dict['P1000023'][2][1] +
       bottleneck_dict['P1000023'][3][1])/4)

print(bottleneck_dict_end['P1000023'][1])

file = open('bottleneck_dict_end.pickle', 'wb')

# dump information to that file
pickle.dump(bottleneck_dict_end, file)

# close the file
file.close()

users, items = [], []
with open("DATA/t1/valid_run.tsv", 'r') as f:
    for line in f:
        linetoks = line.split('\t')
        user_id = linetoks[0]
        item_ids = linetoks[1].strip().split(',')
        for cindex, item_id in enumerate(item_ids):
            users.append(user_id)
            items.append(item_id)

id_bank_dir = f't1_s1-s2-s3_s1-s2-s3-t2_train_all.pickle'
with open(id_bank_dir, 'rb') as centralid_file:
    my_id_bank = pickle.load(centralid_file)

parser = create_arg_parser()

file = open('bottleneck_dict_end.pickle', 'rb')
# dump information to that file
item_emb_in_bottleneck = pickle.load(file)
# close the file
file.close()

exp_names = 'gmf_case'
args = parser.parse_args(f'--exp_name {exp_names} --cuda'.split()) #
if torch.cuda.is_available() and args.cuda:
    torch.cuda.set_device(0)
args.cuda=False
args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print("Device:", args.device)

args.latent_dim = 20

gmf_model = Model(args, my_id_bank)
gmf_model.load("t1_s1-s2-s3_s1-s2-s3-t2_train_all_19.model")

affine_output = torch.nn.Linear(in_features=20, out_features=1)
logistic = torch.nn.Sigmoid()

task_rec_all = []
task_unq_users = set()
cur_users = []

for i in range(len(users)):
  # get user emb. from gmf csv
  user_index = my_id_bank.query_user_index(users[i])
  item_index = my_id_bank.query_item_index(items[i])

  user_index = [user_index]
  item_index = [item_index]

  user_index = torch.LongTensor(user_index)
  item_index = torch.LongTensor(item_index)

  rating_gmf = gmf_model.model(user_index,item_index)
  rating = rating_gmf

  user_emb = gmf_model.model.embedding_user(user_index)
  if items[i] in item_emb_in_bottleneck:
    ae_item_emb = item_emb_in_bottleneck[items[i]]
    element_product = torch.mul(user_emb, torch.FloatTensor(ae_item_emb))
    logits = affine_output(element_product)
    rating_ae = logistic(logits)
    rating = (rating_gmf + rating_ae)/2

  cur_users.append(user_index.item())
  task_rec_all.append((user_index, item_index, rating))
print(task_rec_all[0])
print("a")
task_unq_users = task_unq_users.union(set(cur_users))
print("b")
task_run_mf = get_run_mf(task_rec_all, task_unq_users, my_id_bank)
print("c")
write_run_file(task_run_mf, "ae_gmf_valid.tsv")
print("d")
