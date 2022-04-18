import os
import re
import pickle
import random
import numpy as np
import torch
import nltk
nltk.data.path.append("D:\Program Files\\nltk_data")
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
from gensim.corpora.mmcorpus import MmCorpus
from transformers import BertTokenizer, BertModel

class BertUtils:
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)

    def get_word_vector(self, word, layers=[-3,-2,-1], mode="mean", return_tensor=False):
        '''
        获取BERT词向量
        param: layers: 指定参与计算词向量的隐藏层，-1表示最后一层
        param: mode: 隐藏层合并策略
        return: torch.Tensor, size=[768]
        '''
        output = self.model(**self.tokenizer(word, return_tensors='pt'))
        specified_hidden_states = [output.hidden_states[i] for i in layers]
        specified_embeddings = torch.stack(specified_hidden_states, dim=0)
        # layers to one strategy
        if mode == "sum":
            token_embeddings = torch.squeeze(torch.sum(specified_embeddings, dim=0))
        elif mode == "mean":
            token_embeddings = torch.squeeze(torch.mean(specified_embeddings, dim=0))        
        # tokens to one strategy
        word_embedding = torch.mean(token_embeddings, dim=0)
        if not return_tensor:
            word_embedding = word_embedding.detach().numpy()
        return word_embedding


class Word2VecUtils:
    def __init__(self):
        self.model = Word2Vec.load("/Users/inkding/程序/my-projects/毕设-网易云评论多模态/netease2/models/w2v/c4.mod")
    
    def contains_word(self, word):
        return self.model.wv.__contains__(word)

    def get_word_embedding(self, word):
        return self.model.wv[word]
        

class MyCorpus:
    def __init__(self, source_name, w2v_model_name=None, w2v_model_path=None):
        self.raw_docs = []
        self.source_name = source_name
        self.small_mode = "tfidf"
        # self.small_mode = "df"
        if source_name == "20news":
            self.save_dir = "../data/20news"
            for root, dirs, files in os.walk("../data/20news/raw/20news-bydate"):
                for file in files:
                    if "DS" in file: continue
                    try:
                        self.raw_docs.append(open(os.path.join(root, file), encoding='utf-8').read())
                    except:
                        self.raw_docs.append(open(os.path.join(root, file), encoding='cp1252').read())
        elif source_name == "wiki103":
            self.save_dir = "../data/wiki103"
            for root, dirs, files in os.walk("../data/raw/wikitext-103"):
                for file in files:
                    if "DS" in file: continue
                    content = open(os.path.join(root, file)).read()
                    self.raw_docs.extend(re.split(r'\n =.+= \n', content))
        
        self.stopwords = open("../data/stopwords.txt").read().splitlines()
        
        # 初始化文档 字典 词袋 tfidf模型
        print("总文档数:", len(self.raw_docs))
        self.raw_docs = self.raw_docs
        print("preparing docs...")
        self.set_docs()
        print("preparing dictionary...")
        self.dictionary = Dictionary(self.docs)
        print("字典大小为：", len(self.dictionary))
        self.dictionary.filter_extremes(no_below=20)
        print("过滤后字典大小为：", len(self.dictionary))
        self.dictionary.id2token = {v:k for k,v in self.dictionary.token2id.items()}
        print("preparing bows...")
        self.bows = [self.dictionary.doc2bow(doc) for doc in self.docs]
        print("preparing tfidf model...")
        self.tfidf_model = TfidfModel(self.bows)

        # 通过词频/tfidf缩小文档大小
        print("缩小语料库...")
        self.set_small_docs()
        self.small_dictionary = Dictionary(self.small_docs)
        print("缩小后的字典大小为：", len(self.small_dictionary))
        # 基于BERT模型转换词向量
        self.bert_utils = BertUtils()
        print("转换词向量...")
        self.set_vocab_embeddings()

        self.save()

    def tokenize(self, text):
        words = []
        for word in word_tokenize(text.lower()):
            if len(word) < 2:
                continue
            if word not in self.stopwords and not re.search(r'=|\'|-|`|\.|[0-9]|_|-|~|\^|\*|\\|\||\+', word):
                words.append(word)
        return words


    def set_docs(self):
        self.docs = []
        for doc in self.raw_docs:
            self.docs.append(self.tokenize(doc))

    def get_vocab(self, dictionary=None):
        if not dictionary:
            dictionary = self.dictionary
        return list(dictionary.token2id.keys())

    def _get_small_bow(self, bow, mode="df", topk=20):
        if mode == "df":
            dfs = sorted(bow, key=lambda p:p[1], reverse=True)[:topk]
            return dfs
        elif mode == "tfidf":
            tfidfs = sorted(self.tfidf_model[bow], key=lambda p:p[1], reverse=True)[:topk]
            return tfidfs

    def set_small_docs(self, topk=20):
        '''
        按照词频/tfidf对bow中的词进行重排，并截取topk，获得self.small_docs和self.small_bows
        '''
        self.small_docs = []
        self.small_bows = []
        for i, doc in enumerate(self.docs):
            bow = self._get_small_bow(self.bows[i], mode=self.small_mode, topk=20)
            self.small_bows.append(bow)
            doc = []
            for p in bow:
                doc.append(self.dictionary.id2token[p[0]])
            self.small_docs.append(doc)

    def set_vocab_embeddings(self):    
        self.small_embeddings = []
        for word in self.get_vocab(self.small_dictionary):
            self.small_embeddings.append(self.bert_utils.get_word_vector(word))


    def save(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # 字典
        save_path = os.path.join(self.save_dir, "{}_small_{}_dictionary.pkl".format(self.source_name, self.small_mode))
        with open(save_path, 'wb') as f:
            pickle.dump(self.small_dictionary, f)
        print("已保存字典至:", save_path)
        # 文档
        save_path = os.path.join(self.save_dir, "{}_small_{}_docs.txt".format(self.source_name, self.small_mode))
        with open(save_path, 'w') as f:
            f.write('\n'.join([' '.join(doc) for doc in self.small_docs]))
        print("已保存文档至:", save_path)
        # bow
        save_path = os.path.join(self.save_dir, "{}_small_{}_bows.mm".format(self.source_name, self.small_mode))
        MmCorpus.serialize(save_path, self.small_bows)
        print("已保存bows至:", save_path)
        # 词向量
        save_path = os.path.join(self.save_dir, "{}_small_{}_vocab_embeds.pkl".format(self.source_name, self.small_mode))
        with open(save_path, 'wb') as f:
            pickle.dump(self.small_embeddings, f)
        print("已保存词向量至:", save_path)




if __name__ == "__main__":
    MyCorpus(source_name="20news")
    # MyCorpus(source_name="wiki103")
    # bert_utils = BERTUtils()
    # print(bert_utils.get_word_vector("colonialism").shape)

            

