import torch.utils.data
import argparse
import csv
import sys
import pickle

sys.path.insert(1, 'src')

from src.model import Model
from src.utils import *
from src.data import *

class Datacaller(object):
    def __init__(self, market_name1,market_name2,name_of_pickle1,name_of_pickle2,name_of_ptmodel1,name_of_ptmodel2):
        self.market_name1 = market_name1
        self.market_name2 = market_name2
        self.name_of_pickle1 = name_of_pickle1
        self.name_of_pickle2 = name_of_pickle2
        self.name_of_ptmodel1 = name_of_ptmodel1
        self.name_of_ptmodel2 = name_of_ptmodel2
    def prepare_model_properties(self,market_name,name_of_pickle,name_of_ptmodel):
        def create_arg_parser():
            """Create argument parser for our baseline. """
            parser = argparse.ArgumentParser('GMFbaseline')

            # DATA  Arguments
            parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
            parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t1')
            parser.add_argument('--src_markets',
                                help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training',
                                type=str, default=f'{market_name}')

            parser.add_argument('--tgt_market_valid', help='specify validation run file for target market', type=str,
                                default='DATA/' + f'{market_name}' + '/valid_run.tsv')
            parser.add_argument('--tgt_market_test', help='specify test run file for target market', type=str,
                                default='DATA/t1/test_run.tsv')

            parser.add_argument('--exp_name', help='name the experiment', type=str, default='baseline_toy')

            parser.add_argument('--train_data_file', help='the file name of the train data', type=str,
                                default='train_5core.tsv')  # 'train.tsv' for the original data loading

            # MODEL arguments
            parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
            parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
            parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
            parser.add_argument('--l2_reg', type=float, default=1e-07, help='learning rate')
            parser.add_argument('--latent_dim', type=int, default=20, help='latent dimensions')
            parser.add_argument('--num_negative', type=int, default=4, help='num of negative samples during training')

            parser.add_argument('--cuda', action='store_true', help='use of cuda')
            parser.add_argument('--seed', type=int, default=42, help='manual seed init')

            return parser

        parser = create_arg_parser()
        args = parser.parse_args()
        set_seed(args)

        if torch.cuda.is_available() and args.cuda:
            torch.cuda.set_device(0)
        args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
        print(f'Running experiment on device: {args.device}')

        file = open("checkpoints/" + name_of_pickle, 'rb')

        # dump information to that file
        data = pickle.load(file)
        my_id_bank = data
        mymodel = Model(args, my_id_bank)
        state_dict = mymodel.load("checkpoints/" + name_of_ptmodel)
        return my_id_bank,state_dict

    def prepare_dataLoader(self):
          _,_,item_embedding_1,_,item_embedding_2,intersected_item_count,_ = self.intersection_finder()
          loader1 = torch.utils.data.DataLoader(dataset=item_embedding_1,
                                                  batch_size=intersected_item_count,
                                                  shuffle=False)
          loader2 = torch.utils.data.DataLoader(dataset=item_embedding_2,
                                                  batch_size=intersected_item_count,
                                                  shuffle=False)
          return loader1,loader2

    def intersection_finder(self):
        my_id_bank_1,state_dict_1=self.prepare_model_properties(self.market_name1,self.name_of_pickle1,self.name_of_ptmodel1)
        my_id_bank_2, state_dict_2 = self.prepare_model_properties(self.market_name2,self.name_of_pickle2,self.name_of_ptmodel2)
        s1_train = pd.read_csv("DATA/" + f'{self.market_name1}' + "/train_5core.tsv", sep="\t")
        s2_train = pd.read_csv("DATA/" + f'{self.market_name2}' + "/train_5core.tsv", sep="\t")
        s1_item_list = list(s1_train.itemId.unique())
        s2_item_list = list(s2_train.itemId.unique())
        intersection_of_first_second = [e for e in s1_item_list if e in s2_item_list]
        item_ids = []
        item_embedding_1 = []
        item_indexes_1 = []
        item_embedding_2 = []
        item_indexes_2 = []
        all_info_list = [["id","index1","embedding1","index2","embedding2"],
                         ]
        counter = 1
        for item_id in intersection_of_first_second:
            all_info_list.append([])
            all_info_list[counter].append(item_id)
            item_ids.append(item_id)
            item_index_1 = my_id_bank_1.query_item_index(item_id)
            item_indexes_1.append(item_index_1)
            item_index_2 = my_id_bank_2.query_item_index(item_id)
            item_indexes_2.append(item_index_2)
            item_embedding_s21 = state_dict_1[item_index_1].detach().cpu().numpy()
            item_embedding_s31 = state_dict_2[item_index_2].detach().cpu().numpy()
            item_embedding_1.append(state_dict_1[item_index_1])
            item_embedding_2.append(state_dict_2[item_index_2])
            all_info_list[counter].append(item_index_1)
            all_info_list[counter].append(item_embedding_s21)
            all_info_list[counter].append(item_index_2)
            all_info_list[counter].append(item_embedding_s31)
            counter += 1
        return item_ids,item_indexes_1,item_embedding_1,item_indexes_2,item_embedding_2,counter,all_info_list

    def writer(self):
        list = self.returner()

        with open(f'allInOne{self.market_name1}_{self.market_name2}.tsv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            # write the data
            for i in range(len(list)):
              writer.writerow(list[i])
            print("File has been written")
        return

    def returner(self):
        _, _, _, _, _,_,all_item_list = self.intersection_finder()
        return all_item_list
