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
    '''
    attributes:
        data_source_name: 数据源,如20news,wiki103,...
        dictionary: gensim词典
        docs: 文档集
        vecs: 与词典序号对应的词向量（BERT） 
        bows: 与文档对应bag-of-word表示
    __getitem__:
        return doc, bow
    '''
    def __init__(self, data_source_name) -> None:
        super().__init__()
        self.data_source_name = data_source_name
        
        # 从data文件夹中加载数据
        if data_source_name == "20news":
            data_path = {
                "dict":"../data/20news/20news_small_tfidf_dictionary.pkl",
                "docs":"../data/20news/20news_small_tfidf_docs.txt",
                "bows":"../data/20news/20news_small_tfidf_bows.mm",
                "vecs": "../data/20news/20news_small_tfidf_vocab_embeds.pkl"
            }
        with open(data_path["dict"], 'rb') as f:
            self.dictionary = pickle.load(f)
            self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
        self.docs = open(data_path["docs"]).read().splitlines()
        with open(data_path["vecs"], 'rb') as f:
            self.vecs = pickle.load(f)
        self.bows = MmCorpus(data_path["bows"])
        self.vocab_size = len(self.dictionary)

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocab_size)
        src_bow = self.bows[idx] # [[tokenid, count], ...]
        item = list(zip(*src_bow))
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        bow = torch.where(bow>0, 1.0, 0.0)
        doc = self.docs[idx]
        return doc, bow


    def __len__(self):
        return len(self.docs)
    
    def __iter__(self):
        for doc in self.docs:
            yield doc



def main():
    dataset = MyDataset("20news")
    for i in range(20):
        print(i)
        dataset[i]

if __name__ == "__main__":
    main()