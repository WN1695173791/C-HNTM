from pickletools import optimize
import time
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

from utils import *
from dataset import MyDataset
from HNTM import C_HNTM, C_HNTM_Pretrain

class C_HNTM_Runner:
    def __init__(self, args, device=0, mode="train") -> None:
        '''
        加载参数，模型初始化，预训练聚类模型GMM
        '''
        self.device = get_device(device) # device=-1表示cpu，其他表示gpu序号
        self.save_path = "../models/c_hntm/c_hntm_{}.pkl".format(
                time.strftime("%Y-%m-%d-%H", time.localtime()))
        get_or_create_path(self.save_path)
        
        # 加载数据
        self.data_source_name = args.data
        if self.data_source_name in ("20news", "103wiki"):
            self.dataset = MyDataset(
                data_source_name=self.data_source_name,
        )
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True)
        self.vecs = torch.tensor(np.array(self.dataset.vecs)).to(self.device)

        # 加载参数
        self.args = args
        self.R = args.n_cluster # 顶层主题数
        self.Z = args.n_topic # 底层主题数
        self.vocab_size = self.dataset.vocab_size
        encode_dims = [self.vocab_size, 1024, 512, self.Z] # vae模型结构参数
        decode_dims = [self.Z, 512, 1024, self.vocab_size] # vae模型结构参数
        embed_dim = self.vecs.shape[1]

        self.model = C_HNTM(self.vocab_size, self.R, self.Z, encode_dims, decode_dims, embed_dim)
        self.pretrain_model = C_HNTM_Pretrain(self.vocab_size, self.R, self.Z, encode_dims, decode_dims, embed_dim)
        if mode == "train":
            print("pretrain vae ...")
            self.pretrain_vae()
            print("pretrain gmm ...")
            self.pretrain_GMM() # 预训练GMM模型
            print("train model ...")
            self.model.init_gmm(self.gmm)
            self.model.to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
            

    def pretrain_GMM(self):
        self.gmm = GaussianMixture(n_components=self.R, random_state=21, covariance_type="diag") # 注意方差类型需要设置为diag
        self.gmm.fit(self.dataset.vecs)

    def pretrain_vae(self):
        self.pretrain_model.to(self.device)
        optimizer = torch.optim.SGD(self.pretrain_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for epoch in range(self.args.n_epoch):
            pretrain_epoch_loss = []
            for batch_data in tqdm(self.dataloader):
                optimizer.zero_grad()
                doc, bow = batch_data # doc-文本序列，bow-文档词袋向量
                x = bow
                x = x.to(self.device)
                reconst_x, mu, logvar = self.pretrain_model(x)
                pretrain_loss_dict = self.pretrain_model.loss(x, reconst_x, mu, logvar)
                loss = pretrain_loss_dict["loss"]
                loss.backward()
                optimizer.step()
                pretrain_epoch_loss.append([v.item() for k,v in pretrain_loss_dict.items()])
            
            # wandb log
            pretrain_epoch_loss = np.stack(pretrain_epoch_loss)
            avg_pretrain_epoch_loss = np.mean(pretrain_epoch_loss, axis=0)        
            avg_pretrain_epoch_losses_dict = dict(zip(
                            ["reconst_loss","kld_loss"], avg_pretrain_epoch_loss))     
            wandb_log("pretrain-loss", avg_pretrain_epoch_losses_dict)
            pretrain_metric_dict = self.evaluate(self.pretrain_model)
            wandb_log("pretrain-metric", pretrain_metric_dict)

        # 将pretrain model的参数赋给model
        print("将pretrain model的参数赋给model")
        state_dict = self.pretrain_model.state_dict()
        self.model.load_state_dict(state_dict, strict=False)
            
        

    def train(self):
        for epoch in range(self.args.n_epoch):
            epoch_losses = []
            for batch_data in tqdm(self.dataloader):
                self.optimizer.zero_grad()
                doc, bow = batch_data # doc-文本序列，bow-文档词袋向量
                x = bow
                x = x.to(self.device)
                reconst_x, mu, logvar = self.model(x)
                loss_dict = self.model.loss(x, reconst_x, mu, logvar, self.vecs)
                loss = loss_dict["loss"]
                loss.backward()
                self.optimizer.step()
                epoch_losses.append([v.item() for k, v in loss_dict.items()])

            # loss and metric check
            print("EPOCH-{}".format(epoch))
            # loss
            epoch_losses = np.stack(epoch_losses)
            avg_epoch_losses = np.mean(epoch_losses, axis=0)
            avg_epoch_losses_dict = dict(zip(
                            ["l1_loss","l2_loss","l3_loss","l4_loss","l5_loss","total_loss"], 
                            avg_epoch_losses
                        ))
            wandb_log("loss", avg_epoch_losses_dict)                  
            # metric
            metric_dict = self.evaluate(self.model)
            wandb_log("metric", metric_dict)    

            # if epoch % 10 == 0:
            #     self.show_topic_results()
            

        torch.save(self.model.state_dict(), self.save_path)
        print("model saved to {}".format(self.save_path))

    
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def get_topic_words(self, model):
        '''
        获取底层 topic words
        '''
        # p_matrix_beta = F.softmax(self.model.beta.weight, dim=0).T
        p_matrix_beta = torch.softmax(model.decode(torch.eye(self.Z).to(self.device)), dim=1)
        entropy = -torch.sum(p_matrix_beta * torch.log(p_matrix_beta), axis=1)
        _, top_entropy_topic_leaf_index = torch.topk(entropy, k=20)
        p_matrix_leaf_topic_word = p_matrix_beta[top_entropy_topic_leaf_index]
        _, words_index_matrix = torch.topk(p_matrix_leaf_topic_word, k=10, dim=1)

        topic_words = []
        for words_index in words_index_matrix:
            topic_words.append([self.dataset.dictionary.id2token[i.item()] for i in words_index])        
        return topic_words
        
    
    def show_topic_results(self):
        '''
        展示主题模型结果
        '''
        print("leaf topics ---> words")
        topic_words = self.get_topic_words()
        for words in topic_words:
            print(words)

        print("root topics ---> leaf topics")


    def evaluate(self, model, print=False):
        topic_words = self.get_topic_words(model)
        docs = [line.split() for line in self.dataset.docs]
        topic_coherence_score_dict = calc_topic_coherence(topic_words, docs, self.dataset.dictionary)
        diversity_score = calc_topic_diversity(topic_words)
        metric_dict = topic_coherence_score_dict
        metric_dict["topic_diversity"] = diversity_score

        if print == True:
            for k, v in metric_dict.items():
                print("{}: {:.4f}".format(k, v))
        
        return metric_dict
