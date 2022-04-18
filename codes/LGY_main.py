import os
import sys
import argparse
import torch
from torch.utils import data
from gensim.models import Word2Vec

from run import *
from codes.LGY_dataset import *

parser = argparse.ArgumentParser("NTM")
parser.add_argument("--model", type=str, choices=["gsm", "avitm", "etm", "wtm", "wetm", "vade", "hntm", "my_hntm"])
parser.add_argument("--num_topics", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.01)
parser.add_argument("--w2v_path", type=str, default="/Users/inkding/程序/my-projects/毕设-网易云评论多模态/netease3/models/pretrained/w2v/c4.mod")

args = parser.parse_args()

def main():
    # 加载数据
    dataset = MyDataset(data_source="20news", mode="load")
    if args.model=="gsm":
        runner = NVDM_GSM_Runner(args, dataset)
    elif args.model=="avitm":
        runner = AVITM_Runner(args, dataset)
    elif args.model=="etm":
        runner = ETM_Runner(args, dataset)
    elif args.model=="wtm":
        runner = WTM_Runner(args, dataset)
    elif args.model=="wetm":
        runner = WETM_Runner(args, dataset)
    elif args.model=="vade":
        # ckpt = '.pretrain/vade_pretrain.wght'
        runner = VaDE_Runner(args, dataset)
    elif args.model=="hntm":
        runner = HNTM_Runner(args, dataset)
    elif args.model=="my_hntm":
        dataset = MyClusterDataset(
            data_source="20news",
            mode="load"
        )
        runner = MyHNTM_Runner(args, dataset)
    try:
        runner.train()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt...Terminated.")
        sys.exit(0)
    
    if args.model == "my_hntm":
        with open("../result/20news_word2c_vec_gmm_30.pkl", 'rb') as f:
            cluster_words = pickle.load(f)
        # runner.print_topic_cluster(cluster_words)
        topic_words = runner.get_topic_words()
        for words in topic_words:
            print(words)
        print('='*30)  
        w2v_model = Word2Vec.load(args.w2v_path)
        runner.evaluate(w2v_model)      


    elif args.model != "hntm":
        topic_words = runner.get_topic_words()
        for words in topic_words:
            print(words)
        print('='*30)
        # w2v_model = Word2Vec.load(args.w2v_path)
        # runner.evaluate(w2v_model)


if __name__=="__main__":
    # main()
    dataset = MyClusterDataset(
        data_source="20news",
        mode="load"
    )    
    runner = MyHNTM_Runner(args, dataset)
    runner.model = MyHNTM(
        "avitm",
        encode_dims=[10648, 1024, 512, 100],
        decode_dims=[100, 512, 10648],
        cluster_decode_dims=[10648, 512, 256, 30],
        pretrain_cluster_model_path="../models/pretrained_cluster_model/20news_pretrained_cluster_model_t30_v10648.pkl"
    )
    runner.model.load_state_dict(torch.load("../models/my_hntm/my_hntm_tp100_2022-03-03-12"))
    topic_words = runner.get_topic_words()
    for words in topic_words:
        print(words)
    print('='*30)  
    w2v_model = Word2Vec.load(args.w2v_path)
    runner.evaluate(w2v_model)    
    # dataset = MyClusterDataset(
    #     source_path="../data/docs.txt",
    #     cluster_result_path="../result/word2c_vec_gmm_1.pkl",
    #     mode="load"
    # )    
    # runner = MyHNTM_Runner(args, dataset)
    # runner.model.load_state_dict(torch.load("../models/hntm/hntm_tp100_2022-02-27-08"))
    # w2v_model = Word2Vec.load(args.w2v_path)
    # runner.evaluate(w2v_model)         
