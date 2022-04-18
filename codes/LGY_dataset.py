import os
import pickle
from collections import Counter
from statistics import mode
from sklearn import cluster
import torch
from torch.utils.data import Dataset, DataLoader
import gensim
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
from gensim.models import TfidfModel


class MyDataset(Dataset):
    def __init__(self, data_source, source_path=None, dictionary_path=None, mode="build"):
        '''
        data_source: netease / 20news / wiki103
        source_path[str]: 每一行对应一个文档，一个文档由高频关键词表示，用空格隔开
        mode[str]: build 和 load 两种模式
        '''
        self.data_source = data_source
        self.save_dir = "../models/corpus/{}".format(data_source)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if mode=="build":
            print("Building corpus from scratch...")
            self.docs = [line.split() for line in open(source_path, encoding="utf-8").read().splitlines()]
            
            if not dictionary_path:
                # Dictionary
                self.dictionary = Dictionary(self.docs)
                print("dictionary size:", len(self.dictionary))
                # self.dictionary.filter_extremes(no_below=10)
            else:
                with open(dictionary_path, 'rb') as f:
                    self.dictionary = pickle.load(f)
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
            self.bows = []
            for doc in self.docs:
                self.bows.append(self.dictionary.doc2bow(doc))

            MmCorpus.serialize(os.path.join(self.save_dir, "{}_bows.mm".format(self.data_source)), self.bows)
            with open(os.path.join(self.save_dir, "{}_dictionary.pkl".format(self.data_source)), 'wb') as f:
                pickle.dump(self.dictionary, f)
            with open(os.path.join(self.save_dir, "{}_docs.pkl".format(self.data_source)), 'wb') as f:
                pickle.dump(self.docs, f)            
                
        else: # mode=load
            print("Loading corpus...")
            with open(os.path.join(self.save_dir, "{}_dictionary.pkl".format(self.data_source)), 'rb') as f:
                self.dictionary = pickle.load(f)
            with open(os.path.join(self.save_dir, "{}_docs.pkl".format(self.data_source)), 'rb') as f:
                self.docs = pickle.load(f)
            self.bows = MmCorpus(os.path.join(self.save_dir, "{}_bows.mm".format(self.data_source)))
        self.vocabsize = len(self.dictionary)

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        src_bow = self.bows[idx] # [[tokenid, count], ...]
        item = list(zip(*src_bow))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        doc = ' '.join(self.docs[idx])
        return doc, bow

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        for doc in self.docs:
            yield doc



class MyClusterDataset(Dataset):
    def __init__(self, data_source, source_path=None, cluster_result_path=None, dictionary_path=None, mode="build"):
        '''
        source_path[str]: 每一行对应一个文档，一个文档由高频关键词表示，用空格隔开
        mode[str]: build 和 load 两种模式
        '''
        self.save_dir = "../models/corpus/{}".format(data_source)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.data_source = data_source

        if mode=="build":
            print("Building corpus from scratch...")
            self.docs = [line.split() for line in open(source_path, encoding="utf-8").read().splitlines()]
            print(len(self.docs))
            self.docs = list(filter(None, self.docs))
            print(len(self.docs))
            if not dictionary_path:
            # Dictionary
                self.dictionary = Dictionary(self.docs)
                print("dictionary size:", len(self.dictionary))
            else:
                with open(dictionary_path, 'rb') as f:
                    self.dictionary = pickle.load(f)
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
            self.bows = []
            for doc in self.docs:
                self.bows.append(self.dictionary.doc2bow(doc))                
            
            # cluster_vec
            with open(cluster_result_path, "rb") as f:
                word2cluster_vec = pickle.load(f)
                self.cluster_vecs = []
                self.n_clusters = len(list(word2cluster_vec.values())[0])
                for doc in self.docs:
                    cluster_vec = torch.zeros(self.n_clusters)
                    for word in doc:
                        if word in word2cluster_vec:
                            cluster_vec += torch.Tensor(word2cluster_vec[word])
                    self.cluster_vecs.append(cluster_vec)

            MmCorpus.serialize(os.path.join(self.save_dir, "{}_bows.mm".format(self.data_source)), self.bows)
            with open(os.path.join(self.save_dir, "{}_dictionary.pkl".format(self.data_source)), 'wb') as f:
                pickle.dump(self.dictionary, f)
            with open(os.path.join(self.save_dir, "{}_docs.pkl".format(self.data_source)), 'wb') as f:
                pickle.dump(self.docs, f)
            with open(os.path.join(self.save_dir, "{}_cluster_vecs.pkl".format(data_source)), "wb") as f:
                pickle.dump(self.cluster_vecs, f)
        
        else: # mode=load
            print("Loading corpus...")
            with open(os.path.join(self.save_dir, "{}_dictionary.pkl".format(self.data_source)), 'rb') as f:
                self.dictionary = pickle.load(f)
                self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
            with open(os.path.join(self.save_dir, "{}_docs.pkl".format(self.data_source)), 'rb') as f:
                self.docs = pickle.load(f)
            with open(os.path.join(self.save_dir, "{}_cluster_vecs.pkl".format(self.data_source)), "rb") as f:
                self.cluster_vecs = pickle.load(f)                
            self.bows = MmCorpus(os.path.join(self.save_dir, "{}_bows.mm".format(self.data_source)))
        self.vocabsize = len(self.dictionary)

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        src_bow = self.bows[idx] # [[tokenid, count], ...]
        item = list(zip(*src_bow))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        doc = ' '.join(self.docs[idx])
        cluster_vec = self.cluster_vecs[idx]
        return doc, bow, cluster_vec

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        for doc in self.docs:
            yield doc
        



if __name__=="__main__":
    # test_dataset = MyDataset(
    #     data_source="netease",
    #     source_path="../data/netease_docs.txt",
    #     mode="build"
    # )
    test_dataset = MyClusterDataset(
        data_source="netease",
        source_path="../data/netease_docs.txt",
        cluster_result_path="../result/netease_word2c_vec_gmm_1.pkl",
        mode="build"
    )
    # for i in range(10):
    #     doc, bow, cluster_vec = test_dataset[i]
    #     print(doc)
    #     print(bow.shape)
    #     print(cluster_vec.shape)
    
            



