import os
import pickle
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import gensim
from gensim.corpora.mmcorpus import MmCorpus
from gensim.corpora import Dictionary
from gensim.models import TfidfModel


class MyDataset(Dataset):
    def __init__(self, source_path, mode="build"):
        '''
        source_path[str]: 每一行对应一个文档，一个文档由高频关键词表示，用空格隔开
        mode[str]: build 和 load 两种模式
        '''
        self.save_dir = "models/corpus"
        if mode=="build":
            print("Building corpus from scratch...")
            self.docs = [line.split() for line in open(source_path, encoding="utf-8").read().splitlines()]
            # Dictionary
            self.dictionary = Dictionary(self.docs)
            self.dictionary.filter_extremes(no_below=10)
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
            # BoWs
            self.bows = []
            new_docs = [] # dictionary过滤之后可能有的文档会变成空
            for doc in self.docs:
                bow_ = self.dictionary.doc2bow(doc)
                if bow_:
                    self.bows.append(bow_)
                    new_docs.append(doc)
            self.docs = new_docs

            MmCorpus.serialize(os.path.join(self.save_dir, "bows.mm"), self.bows)
            with open(os.path.join(self.save_dir, "dictionary.pkl"), 'wb') as f:
                pickle.dump(self.dictionary, f)
            with open(os.path.join(self.save_dir, "docs.pkl"), 'wb') as f:
                pickle.dump(self.docs, f)
        else: # mode=load
            print("Loading corpus...")
            with open(os.path.join(self.save_dir, "dictionary.pkl"), 'rb') as f:
                self.dictionary = pickle.load(f)
            with open(os.path.join(self.save_dir, "docs.pkl"), 'rb') as f:
                self.docs = pickle.load(f)
            self.bows = MmCorpus(os.path.join(self.save_dir, "bows.mm"))
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



if __name__=="__main__":
    test_dataset = MyDataset(source_path="data/docs.txt")
    for i in range(10):
        doc, bow = test_dataset[i]
        print(doc)
        print(bow.shape)
    
            



