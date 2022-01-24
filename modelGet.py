
import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import csv
import os
import json
"""import resource"""
import sys
import pickle

sys.path.insert(1, 'src')

from src.model import Model
from src.utils import *
from src.data import *

from src.utils import read_qrel_file ,get_evaluations_final




def create_arg_parser() :
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('GMFbaseline')

    # DATA  Arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t1')
    parser.add_argument('--src_markets', help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training', type=str, default='s1')

    parser.add_argument('--tgt_market_valid', help='specify validation run file for target market', type=str, default='DATA/s1/valid_run.tsv')
    parser.add_argument('--tgt_market_test', help='specify test run file for target market', type=str, default='DATA/t1/test_run.tsv')

    parser.add_argument('--exp_name', help='name the experiment' ,type=str, default='baseline_toy')

    parser.add_argument('--train_data_file', help='the file name of the train data' ,type=str, default='train_5core.tsv')  # 'train.tsv' for the original data loading


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



def main():

    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Running experiment on device: {args.device}')

    ############
    ## Target Market data
    ############
    my_id_bank = Central_ID_Bank()

    train_file_names = args.train_data_file # 'train_5core.tsv', 'train.tsv' for the original data loading

    tgt_train_data_dir = os.path.join(args.data_dir, args.tgt_market, train_file_names)
    tgt_train_ratings = pd.read_csv(tgt_train_data_dir, sep='\t')

    print(f'Loading target market {args.tgt_market}: {tgt_train_data_dir}')
    tgt_task_generator = TaskGenerator(tgt_train_ratings, my_id_bank)
    print('Loaded target data!\n')

    # task_gen_all: contains data for all training markets, index 0 for target market data
    task_gen_all = {
        #0: tgt_task_generator
    }

    ############
    ## Source Market(s) Data
    ############
    src_market_list = args.src_markets.split('-')
    if 'none' not in src_market_list:
        cur_task_index = 1
        for cur_src_market in src_market_list:
            cur_src_data_dir = os.path.join(args.data_dir, cur_src_market, train_file_names)
            print(f'Loading {cur_src_market}: {cur_src_data_dir}')
            cur_src_train_ratings = pd.read_csv(cur_src_data_dir, sep='\t')
            cur_src_task_generator = TaskGenerator(cur_src_train_ratings, my_id_bank)
            task_gen_all[cur_task_index] = cur_src_task_generator
            cur_task_index +=1

    train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=args.num_negative, meta_split='train' )

    ############
    ## Validation and Test Run
    ############
    tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_valid, args.batch_size)
    tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_test, args.batch_size)

    ############
    ## Model
    ############
    file = open("checkpoints/" + "t1_s1_baseline_toy.pickle", 'rb')

    # dump information to that file
    data = pickle.load(file)
    mymodel = Model(args, data)
    c = 0
    state_dict = mymodel.load("checkpoints/" + "t1_s1_baseline_toy.model")
    for i in state_dict:
         print(c)
         print(i)
         c += 1
    return state_dict

main()