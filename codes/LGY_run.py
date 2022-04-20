import os
import time
from tqdm import tqdm
import itertools
import numpy as np
from gensim.models import Word2Vec
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.optim import SGD
from utils import *
from models import *
from dataset import *
from HNTM import C_HNTM, HNTM, MyHNTM

class Runner:
    def __init__(self, args, dataset, model_name, mode="train"):
        # 加载config
        self.NUM_EPOCHS = args.num_epochs
        self.BATCH_SIZE = args.batch_size
        self.LR = args.lr
        self.NUM_TOPICS = args.num_topics
        self.DROPOUT = args.dropout
        # self.EMBED_DIM = args.embed_dim

        # 加载数据
        self.dataset = dataset
        self.vocab_size = dataset.vocabsize
        self.docs = self.dataset.docs
        self.dictionary = self.dataset.dictionary
        self.dataloader =  DataLoader(
            self.dataset, batch_size=self.BATCH_SIZE, shuffle=True
        )
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建路径
        self.model_name = model_name
        self.save_dir = "../models/{}".format(self.model_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir,
            "{}_tp{}_{}".format(self.model_name, self.NUM_TOPICS, time.strftime("%Y-%m-%d-%H", time.localtime()))
        )


    def train(self):
        pass
    
    def evaluate(self, w2v_model=None):
        topic_words = self.get_topic_words()
        # topic coherence
        topic_coherence_score = calc_topic_coherence(topic_words, self.docs, self.dictionary)
        for k, v in topic_coherence_score.items():
            print("{}: {:.4f}".format(k, v[0]))

        # intra-topic distance
        intra_topic_simi = calc_intra_topic_similarity(topic_words, w2v_model)
        print("intra-topic similarity: {:.4f}".format(intra_topic_simi))
        
        # topic_diversity
        print("topic diversity: {:.4f}".format(calc_topic_diversity(topic_words)))

    def get_topic_words(self, topk=10):
        '''
        获取每个主题下的 topk words。
        '''
        topic_words = []
        idxes = torch.eye(self.NUM_TOPICS).to(self.device)
        word_dist = torch.softmax(self.model.decode(idxes), dim=1)
        vals, indices = torch.topk(word_dist, topk, dim=1)
        vals.cpu().tolist()
        indices.cpu().tolist() # 将tensor数据类型转换为基本数据类型
        for i in range(self.NUM_TOPICS):
            topic_words.append([self.dictionary.id2token[int(idx)] for idx in indices[i]])
        return topic_words




class NVDM_GSM_Runner(Runner):
    def __init__(self, args, dataset):
        super(NVDM_GSM_Runner, self).__init__(args, dataset, model_name="gsm")
        self.model = NVDM_GSM(
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            decode_dims=[self.NUM_TOPICS, 512, self.vocab_size],
            dropout=self.DROPOUT
        )
        self.model.to(self.device)


    def train(self):
        self.model.train() 
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR, momentum=0.9)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, mu, logvar = self.model(x)
                loss = self.model.loss(x, x_reconst, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)




class AVITM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, model_name="avitm", mode=mode)
        self.model = AVITM(
            encode_dims=[self.vocab_size, 512, self.NUM_TOPICS],
            decode_dims=[self.NUM_TOPICS, 512, self.vocab_size],
            dropout=self.DROPOUT
        )
        self.model.to(self.device)
        self.params = self.model.parameters()
        self.optimizer = torch.optim.SGD(self.params, lr=self.LR)
        # self.scheduler = CyclicLR(self.optimizer, base_lr=1e-3, max_lr=1e-2, step_size_up=200, step_size_down=200)


    def train(self):
        self.model.train()
        
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, mu, logvar = self.model(x)
                loss = self.model.loss(x, x_reconst, mu, logvar)

                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item()/len(bows))
                # self.scheduler.step()

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)        



class ETM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, model_name="etm", mode=mode)
        
        # 载入w2v模型
        w2v_model = Word2Vec.load(args.w2v_path)
        words = list(self.dictionary.token2id.keys())
        rho_init = []
        for w in words:
            if w2v_model.wv.__contains__(w):
                rho_init.append(w2v_model.wv.get_vector(w))
            else:
                rho_init.append(np.zeros(w2v_model.vector_size))
        rho_init = torch.Tensor(rho_init)
        print(rho_init.shape)

        self.model = ETM(
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            vocab_size=self.vocab_size,
            embed_dim=300,
            rho_init=rho_init,
            dropout=self.DROPOUT
        )
        self.model.to(self.device)

        rho_param_ids = list(map(id, self.model.rho.parameters()))
        base_params = filter(lambda p:id(p) not in rho_param_ids, self.model.parameters())
        self.params = [
            {"params": self.model.rho.parameters(), "lr": 1e-5}, 
            {"params": base_params, "lr":self.LR}
        ]

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.params, lr=self.LR, momentum=0.9)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, mu, logvar = self.model(x)
                loss = self.model.loss(x, x_reconst, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)             



class WTM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, "wtm", mode=mode)
        self.model = WAE(
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            decode_dims=[self.NUM_TOPICS, 512, self.vocab_size],
            dropout=self.DROPOUT
        )
        self.model.to(self.device)

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, theta = self.model(x)
                theta_prior = self.model.sample(dist="dirichlet", batch_size=len(x), ori_data=x).to(self.device)
                
                loss = self.model.loss(x, x_reconst, theta_prior, theta, device=self.device, beta=1.0)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)              



class WETM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, "wetm", mode=mode)

        # 加载预训练词向量
        w2v_model = Word2Vec.load(args.w2v_path)
        words = list(self.dictionary.token2id.keys())
        rho_init = []
        for w in words:
            if w2v_model.wv.__contains__(w):
                rho_init.append(w2v_model.wv.get_vector(w))
            else:
                rho_init.append(np.zeros(w2v_model.vector_size))
        rho_init = torch.Tensor(rho_init)
        print(rho_init.shape)        

        self.model = WETM(
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            embed_dim=300,
            vocab_size = self.vocab_size,
            rho_init=rho_init,
            dropout=self.DROPOUT
        )
        self.model.to(self.device)


        rho_param_ids = list(map(id, self.model.rho.parameters()))
        base_params = filter(lambda p:id(p) not in rho_param_ids, self.model.parameters())
        self.params = [
            {"params": self.model.rho.parameters(), "lr": 1e-5}, 
            {"params": base_params, "lr":self.LR}
        ]

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.params, lr=self.LR, momentum=0.9)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, theta = self.model(x)
                theta_prior = self.model.sample(dist="dirichlet", batch_size=len(x), ori_data=x).to(self.device)
                
                loss = self.model.loss(x, x_reconst, theta_prior, theta, device=self.device, beta=1.0)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)


class VaDE_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, "vade", mode=mode)
        self.model = VaDE(
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            decode_dims=[self.NUM_TOPICS, 512, self.vocab_size],
            dropout=self.DROPOUT,
            n_clusters=10
        )
        self.model.to(self.device)
    
    def pretrain(self, dataloader, retrain=True, pre_epoch=30):
        if (not os.path.exists('.pretrain/vade_pretrain.wght')) or retrain==True:
            if not os.path.exists('.pretrain/'):
                os.mkdir('.pretrain')
            optimizer = torch.optim.SGD(itertools.chain(self.model.encoder.parameters(),\
                self.model.fc_mu.parameters(),\
                    self.model.fc1.parameters(),\
                        self.model.decoder.parameters()), lr=0.005)

            print('Start pretraining ...')
            self.model.train()
            for epoch in tqdm(range(pre_epoch)):
                total_loss = []
                n_instances = 0
                for data in dataloader:
                    optimizer.zero_grad()
                    txts, bows = data
                    bows = bows.to(self.device)
                    bows_recon,_mus,_log_vars = self.model(bows,collate_fn=lambda x: F.softmax(x,dim=1),isPretrain=True)
                    #bows_recon,_mus,_log_vars = self.vade(bows,collate_fn=None,isPretrain=True)
                    logsoftmax = torch.log_softmax(bows_recon,dim=1)
                    rec_loss = -1.0 * torch.sum(bows*logsoftmax)
                    rec_loss /= len(bows)
                    
                    rec_loss.backward()
                    optimizer.step()
                    total_loss.append(rec_loss.item())
                    n_instances += len(bows)
                print("Pretrain: epoch:{:03d}\taverage_loss:{:.3f}".format(epoch, sum(total_loss)/n_instances))
            self.model.fc_logvar.load_state_dict(self.model.fc_mu.state_dict())
            print('Initialize GMM parameters ...')
            z_latents = torch.cat([self.model.get_latent(bows.to(self.device)) for txts,bows in tqdm(dataloader)],dim=0).detach().cpu().numpy()
            # TBD_corvarance_type
            try:
                self.model.gmm.fit(z_latents)

                self.model.pi.data = torch.from_numpy(self.model.gmm.weights_).to(self.device).float()
                self.model.mu_c.data = torch.from_numpy(self.model.gmm.means_).to(self.device).float()
                self.model.logvar_c.data = torch.log(torch.from_numpy(self.model.gmm.covariances_)).to(self.device).float()
            except:
                self.model.mu_c.data = torch.from_numpy(np.random.dirichlet(alpha=1.0*np.ones(self.model.n_clusters)/self.model.n_clusters,size=(self.model.n_clusters,self.model.latent_dim))).float().to(self.device)
                self.model.logvar_c.data = torch.ones(self.model.n_clusters,self.model.latent_dim).float().to(self.device)

            torch.save(self.model.state_dict(),'.pretrain/vade_pretrain.wght')
            print('Store the pretrain weights at dir .pretrain/vade_pretrain.wght')

        else:
            self.vade.load_state_dict(torch.load('.pretrain/vade_pretrain.wght'))



    def train(self):
        self.model.train()
        self.pretrain(self.dataloader, pre_epoch=30, retrain=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.LR)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, mu, logvar = self.model(x)
                loss = self.model.loss(x, x_reconst, mu, logvar)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)    


class HNTM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, "hntm", mode=mode)
        self.topic_dims = [100, 50, 10]
        self.model = HNTM(
            encode_dims=[self.vocab_size, 1024, 512, 300],
            topic_dims=self.topic_dims,
            n_hidden=300,
            dropout=self.DROPOUT,
        )
        self.model.to(self.device)
        self.params = self.model.parameters()

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.params, lr=self.LR, momentum=0.9)
        global_loss = [] # 全局纪录
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst, mu, logvar, depends = self.model(x)
                
                loss = self.model.loss(x, x_reconst, mu, logvar, depends)

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)
    
    def get_topic_words(self, topk=10):
        for i in range(len(self.topic_dims)):
            print("层次[{}]".format(i))
            topic_words = []
            idxes = torch.eye(self.topic_dims[i]).to(self.device)
            beta = self.model.beta_topics[i](self.model.beta_word.weight).transpose(0,1)
            word_dist = torch.softmax(torch.matmul(idxes, beta), dim=1)
            vals, indices = torch.topk(word_dist, topk, dim=1)
            vals.cpu().tolist()
            indices.cpu().tolist() # 将tensor数据类型转换为基本数据类型
            for i in range(self.topic_dims[i]):
                topic_words.append([self.dictionary.id2token[int(idx)] for idx in indices[i]])
            for j, words in enumerate(topic_words):
                print("层次[{}]主题[{}]:".format(i,j), end=" ")
                print(words)
            print("="*30)

    
class MyHNTM_Runner(Runner):
    def __init__(self, args, dataset, mode="train"):
        super().__init__(args, dataset, "my_hntm", mode=mode)
        # self.topic_model_name = "wae"
        self.topic_model_name = "avitm"
        n_clusters=30
        self.model = MyHNTM(
            topic_model_name=self.topic_model_name,
            encode_dims=[self.vocab_size, 1024, 512, self.NUM_TOPICS],
            decode_dims=[self.NUM_TOPICS, 512, self.vocab_size],
            cluster_decode_dims=[self.vocab_size, 512, 256, n_clusters],
            pretrain_cluster_model_path="../models/pretrained_cluster_model/20news_pretrained_cluster_model_t30_v10648.pkl",
            dropout=self.DROPOUT
        )
        self.model.to(self.device)
        topic_model_param_ids = list(map(id, self.model.topic_model.parameters()))
        base_params = filter(lambda p:id(p) not in topic_model_param_ids, self.model.parameters())
        self.params = [
            {"params": self.model.topic_model.parameters(), "lr": 1e-3}, 
            {"params": base_params, "lr":self.LR}
        ]
        self.optimizer = torch.optim.SGD(self.params, lr=self.LR, momentum=0.9)
        # self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 20, 30, 40], gamma=0.1, last_epoch=-1, verbose=True)
        self.topic_words = None

    def train(self):
        self.model.train()
        for epoch in range(self.NUM_EPOCHS):
            # epoch loss
            epoch_loss = []
            epoch_topic_loss = []
            epoch_cluster_loss = []
            epoch_structure_loss = []
            for iter, data in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                doc, bows, cluster_vec = data
                x = bows
                y = cluster_vec
                x.to(self.device)
                y.to(self.device)
                if self.topic_model_name in ("gsm", "avitm"):
                    x_reconst, y_pred, mu, logvar = self.model(x)
                    loss, topic_loss, cluster_loss, structure_loss = self.model.loss(x, x_reconst, y, y_pred, mu=mu, logvar=logvar)
                elif self.topic_model_name == "wae":
                    x_reconst, y_pred, theta = self.model(x)
                    theta_prior = self.model.topic_model.sample(dist="dirichlet", batch_size=len(x), ori_data=x).to(self.device)
                    loss, topic_loss, cluster_loss, structure_loss = self.model.loss(x, x_reconst, y, y_pred, 
                                                    theta_prior=theta_prior, theta_post=theta, 
                                                    device=self.device, beta=1.0)

                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item()/len(bows))
                epoch_topic_loss.append(topic_loss.item()/len(bows))
                epoch_cluster_loss.append(cluster_loss.item()/len(bows))
                epoch_structure_loss.append(structure_loss.item()/len(bows))

            # 调整学习率
            # self.scheduler.step()
            if epoch<10:
                self.optimizer.param_groups[0]["lr"] = 1e-3
            else:
                 self.optimizer.param_groups[0]["lr"] = 1e-4
            print("model.topic_model parameters' lr:", self.optimizer.param_groups[0]["lr"])

            print("Epoch {} AVG Loss: {:.6f}, topic: {:.6f}, cluster: {:.6f}, structure: {:.6f}".format(
                epoch+1, sum(epoch_loss)/len(epoch_loss), 
                sum(epoch_topic_loss)/len(epoch_topic_loss), 
                sum(epoch_cluster_loss)/len(epoch_cluster_loss),
                sum(epoch_structure_loss)/len(epoch_structure_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)

        torch.save(self.model.state_dict(), self.save_path)
        print("Model saved. path: " + self.save_path)
    
    def get_topic_words(self, topk=10):
        topic_words = []
        idxes = torch.eye(self.NUM_TOPICS).to(self.device)
        word_dist = torch.softmax(self.model.topic_model.decode(idxes), dim=1)
        vals, indices = torch.topk(word_dist, topk, dim=1)
        vals.cpu().tolist()
        indices.cpu().tolist() # 将tensor数据类型转换为基本数据类型
        for i in range(self.NUM_TOPICS):
            topic_words.append([self.dictionary.id2token[int(idx)] for idx in indices[i]])
        self.topic_words = topic_words
        return topic_words
        
    def print_topic_cluster(self, cluster_words):
        if not self.topic_words:
            self.get_topic_words()
        idxes = torch.eye(self.NUM_TOPICS).to(self.device)
        out = self.model.cluster_decode(F.softmax(self.model.topic_model.decode(idxes), dim=1))
        cluster_dist = torch.softmax(out, dim=1).detach().numpy()
        cluster_ids = np.argmax(cluster_dist, axis=1)
        for i in range(self.NUM_TOPICS):
            print("===== Topic_{} =====".format(i))
            print("cluster words:", cluster_words[cluster_ids[i]][:15])
            print("topic_words:", self.topic_words[i])
        




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
        self.K = args.n_topic # 底层主题数
        self.vocab_size = self.dataset.vocab_size
        encode_dims = [self.vocab_size, 1024, 512, self.K] # vae模型结构参数
        embed_dim = self.vecs.shape[1]

        self.model = C_HNTM(self.vocab_size, self.R, self.K, encode_dims, embed_dim)
        if mode == "train":
            self.pretrain_GMM() # 预训练GMM模型
            print("GMM模型预训练完成")
            self.model.init_gmm(self.gmm)
            self.model.to(self.device)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
            

    def pretrain_GMM(self):
        self.gmm =  GaussianMixture(n_components=self.R, random_state=21, covariance_type="diag") # 注意方差类型需要设置为diag
        self.gmm.fit(self.dataset.vecs)

    def train(self):
        for epoch in range(self.args.n_epoch):
            epoch_loss = []
            for batch_data in tqdm(self.dataloader):
                self.optimizer.zero_grad()
                doc, bow = batch_data # doc-文本序列，bow-文档词袋向量
                x = bow
                x = x.to(self.device)
                reconst_x, mu, logvar = self.model(x)
                loss = self.model.loss(x, reconst_x, mu, logvar, self.vecs)
                loss.backward()
                self.optimizer.step()
                epoch_loss.append(loss.item()) # 按batch_size取平均
            print("epoch-{} AVG loss:{:.6f}".format(epoch, np.mean(epoch_loss)))
            if epoch % 10 == 0:
                self.evaluate()
                self.show_topic_results()

        torch.save(self.model.state_dict(), self.save_path)
        print("model saved to {}".format(self.save_path))

    
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def get_topic_words(self):
        '''
        给self.topic_words赋值
        '''
        p_matrix_beta = F.softmax(self.model.beta.weight, dim=0).T
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


    def evaluate(self):
        topic_words = self.get_topic_words()
        # topic coherence
        print("开始计算topic-coherence...")
        docs = [line.split() for line in self.dataset.docs]

        npmi_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=self.dataset.dictionary, coherence="c_npmi")
        print("模型构建完成，开始计算")
        coherence_score = npmi_model.get_coherence()

        # topic_coherence_score = calc_topic_coherence(topic_words, docs, self.dataset.dictionary)
        # for k, v in topic_coherence_score.items():
        #     print("{}: {:.4f}".format(k, v[0]))

        # topic_diversity
        print("topic diversity: {:.4f}".format(calc_topic_diversity(topic_words)))        
