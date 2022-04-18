import torch
import torch.nn as nn 
import torch.nn.functional as F
from utils import * 
from models import *
from pretrain_cluster_model import *


class HNTM(nn.Module):
    def __init__(self, encode_dims, topic_dims, n_hidden, dropout=0.01):
        '''
        topic_dims: 层级主题维度列表，如[300, 100, 20]，从叶子到根。
        '''
        super().__init__()
        self.vocab_size = encode_dims[0]
        self.topic_dims = topic_dims
        self.encoder = nn.ModuleDict({
            "enc_{}".format(i): nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        # 中间向量正式进入层级结构的底层
        self.fc_theta3 = nn.Linear(encode_dims[-1], topic_dims[0])

        # depends(D)
        self.depend_vecs1 = nn.ModuleList([
            nn.Linear(n_hidden, topic_dims[i]) for i in range(len(topic_dims)-1)
        ])
        self.depend_vecs2 = nn.ModuleList([
            nn.Linear(n_hidden, topic_dims[i]) for i in range(1, len(topic_dims))
        ])
        # self.depends = nn.ModuleList([
        #     nn.Linear(topic_dims[i], topic_dims[i+1]) for i in range(len(topic_dims)-1)
        # ])

        # beta
        self.beta_topics = nn.ModuleList([
            nn.Linear(n_hidden, topic_dims[i]) for i in range(len(topic_dims))
        ])
        self.beta_word = nn.Linear(n_hidden, self.vocab_size)

        # weight
        self.module_weight = nn.ModuleList([
            nn.Linear(encode_dims[-2], n_hidden),
            nn.Linear(n_hidden, len(topic_dims))
        ])

        self.dropout = nn.Dropout(p=dropout)


    def encode(self, x):
        hid = x
        for _, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, logvar = self.fc_mu(hid), self.fc_logvar(hid)

        weights = hid

        for layer in self.module_weight:
            weights = F.relu(self.dropout(layer(weights)))
        return mu, logvar, weights

    def reparameterize(self, mu, logvar):
        eps = torch.rand_like(mu)
        std = torch.exp(logvar/2)
        z = mu + eps*std
        return z
    
    def decode(self, hid, weights):
        hid = F.relu(self.dropout(self.fc_theta3(hid)))
        x_reconst = torch.zeros(hid.shape[0], self.vocab_size)
        depends = []
        for i in range(len(self.topic_dims)):
            beta = self.beta_topics[i](self.beta_word.weight).transpose(1, 0)
            logits = torch.matmul(hid, beta)
            x_reconst += logits * weights[:,i].reshape(weights[:,i].shape[0],1)

            if i<len(self.topic_dims)-1:
                depend = F.softmax(self.depend_vecs1[i](self.depend_vecs2[i].weight), dim=0).transpose(0,1)
                # depend = F.softmax(torch.matmul(self.depend_vecs1[i], self.depend_vecs2[i]))
                depends.append(depend)
                hid = torch.matmul(hid, depend)
        return x_reconst, depends

    
    def forward(self, x):
        mu, logvar, weights = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst, depends = self.decode(z, weights)
        return x_reconst, mu, logvar, depends     


    def loss(self, x, x_reconst, mean_post, logvar_post, depends, kl_weight=0.1, discrete_weight=0.1, balance_weight=0.01, manifold_weight=None):
        # 重构损失
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax)
        kld_loss = -0.5 * torch.sum(1+logvar_post-mean_post.pow(2)-logvar_post.exp()) # 分布约束正则项
        discrete_loss = 1*(torch.norm(depends[0],p=2) + torch.norm(depends[1],p=2))
        balance_loss = torch.norm(torch.sum(depends[0]), p=2) + torch.norm(torch.sum(depends[1],0), p=2)
        loss = rec_loss + kl_weight*kld_loss  \
                         - discrete_weight*discrete_loss + balance_weight*balance_loss\
                        # + manifold_weight*manifold_loss
        return loss



class MyHNTM(nn.Module):
    def __init__(self, 
        topic_model_name, 
        encode_dims, decode_dims, cluster_decode_dims, 
        pretrain_cluster_model_path, 
        dropout=0.01):
        super().__init__()
        # 传统ntm结构
        self.topic_model_name = topic_model_name
        self.cluster_decode_dims = cluster_decode_dims
        self.num_topics = encode_dims[-1]
        if topic_model_name == "gsm":
            self.topic_model = NVDM_GSM(encode_dims, decode_dims)
        elif topic_model_name == "avitm":
            self.topic_model = AVITM(encode_dims, decode_dims)
        elif topic_model_name == "wae":
            self.topic_model = WAE(encode_dims, decode_dims)
        
        self.cluster_decoder = nn.ModuleList([
            nn.Linear(cluster_decode_dims[i], cluster_decode_dims[i+1])
            for i in range(len(cluster_decode_dims)-1)
        ])
        self. init_cluster_decoder(pretrain_cluster_model_path)
        self.dropout = nn.Dropout(dropout)

    def init_cluster_decoder(self, pretrain_cluster_model_path):
        cluster_model = ClusterModel(self.cluster_decode_dims)
        cluster_model.load_state_dict(torch.load(pretrain_cluster_model_path))
        for i in range(len(self.cluster_decode_dims)-1):
            self.cluster_decoder[i].weight = nn.Parameter(cluster_model.module[i].weight)
 

    def cluster_decode(self, hid):
        for i in range(len(self.cluster_decoder)):
            hid = F.relu(self.dropout(self.cluster_decoder[i](hid)))
        return hid
        
    def forward(self, x):
        if self.topic_model_name in ("gsm", "avitm"):
            x_reconst, mu, logvar = self.topic_model(x)
            out = self.cluster_decode(F.softmax(x_reconst, dim=1))
            return x_reconst, out, mu, logvar
        elif self.topic_model_name == "wae": 
            x_reconst, theta = self.topic_model(x)
            out = self.cluster_decode(F.softmax(x_reconst, dim=1))
            return x_reconst, out, theta

    def get_structure_loss(self):
        idxes = torch.eye(self.num_topics)
        out = self.cluster_decode(F.softmax(self.topic_model.decode(idxes), dim=1))
        cluster_dist = torch.softmax(out, dim=1)
        loss = -torch.sum(torch.square(cluster_dist))
        return loss

    
    def loss(self, x, x_reconst, y, y_pred, 
                mu=None, logvar=None, 
                theta_prior=None, theta_post=None, device=None, beta=None):
        a, b, c = 0.5, 0.2, 0.3
        if self.topic_model_name in ("gsm", "avitm"):
            loss_topic = self.topic_model.loss(x, x_reconst, mu, logvar)
        elif self.topic_model_name == "wae":
            loss_topic = self.topic_model.loss(x, x_reconst, theta_prior, theta_post, device, beta)
        loss_cluster = -1 * torch.sum(y * torch.log_softmax(y_pred, dim=1))
        structure_loss = self.get_structure_loss()
        loss = (a*loss_topic + b*loss_cluster + c*structure_loss) * 3
        return loss, loss_topic, loss_cluster, structure_loss





class C_HNTM(nn.Module):
    def __init__(self, vocab_size, n_topic_root, n_topic_leaf, encode_dims, embed_dim):
        '''
        基于VaDE改造的层次神经网络。
        数据流：
        >> x:(batch_size, vocab_ size) 
        => encoder
        >> mu:(batch_size, n_topic_leaf), logvar:(batch_size, n_topic_leaf)
        => reparametize
        => decoder 
        >> z:(batch_size, n_topic_leaf) 
        >> reconst_x:(batch_size, vocab_size)
        '''
        super(C_HNTM, self).__init__()

        # nn.Parameter 参与参数优化
        self.gmm_pi = nn.Parameter(torch.zeros(n_topic_root)) # gmm param
        self.gmm_mu = nn.Parameter(torch.randn(n_topic_root, embed_dim)) # gmm param
        self.gmm_logvar = nn.Parameter(torch.randn(n_topic_root, embed_dim)) # gmm param

        self.encoder = nn.ModuleList([
            nn.Linear(encode_dims[i], encode_dims[i+1]) for i in range(len(encode_dims)-2)
        ])

        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])

        self.decoder = None
        self.beta = nn.Linear(n_topic_leaf, vocab_size)
        self.alpha = nn.Linear(n_topic_root, n_topic_leaf)

        self.dropout = nn.Dropout(p=0.01)
    
    def init_gmm(self, gmm):
        '''
        由GMM模型初始化参数
        '''
        self.gmm_mu.data = torch.from_numpy(gmm.means_).float()
        self.gmm_logvar.data = torch.log(torch.from_numpy(gmm.covariances_)).float()
        self.gmm_pi.data = torch.log(torch.from_numpy(gmm.weights_)).float()


    def reparameterize(self, mu, logvar):
        '''
        重参方法，基于encoder给出的期望和对数方差生成z。
        '''
        # torch.randn_like(input): 返回和input大小一致且服从正态分布的tensor
        eps = torch.randn_like(mu)
        std = torch.exp(logvar/2)
        z = mu + eps*std
        return z

    @property
    def weights(self):
        return torch.softmax(self.gmm_pi, dim=0)

    def encode(self, x):
        hid = x
        for i in range(len(self.encoder)):
            hid = F.relu(self.dropout(self.encoder[i](hid)))
        mu = self.fc_mu(hid)
        logvar = self.fc_logvar(hid)
        return mu, logvar

    def decode(self, z):
        weight = self.beta.weight
        weight = F.softmax(weight, dim=0).transpose(1,0)
        x_reconst = torch.mm(z, weight)
        return x_reconst

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss(self, x, x_reconst, mu, logvar, vecs):
        '''
        x: size=(batch_size, vocab_size)
        mu: size=(batch_size, n_topic_leaf)
        logvar: size=(batch_size, n_topic_leaf)
        vecs: size=(vocab_size, embed_dim)
        '''
        alpha = F.softmax(self.alpha.weight, dim=0) # size=(n_topic_leaf, n_topic_root)
        beta = F.softmax(self.beta.weight, dim=0) # size=(vocab_size, n_topic_leaf)

        # self.gmm_mu size=(n_topic_root, embed_dim)
        # self.gmm_logvar size=(n_topic_root, embed_dim)
        # self.gmm_pi size=(n_topic_root)

        # q(z_i|x)
        # https://stats.stackexchange.com/questions/321947/expectation-of-the-softmax-transform-for-gaussian-multivariate-variables
        c = 1. / torch.sqrt(1 + torch.exp(logvar))
        gamma = 1 - 1 / (1 + torch.exp(mu * c)) # gamma size=(batch_size, n_topic_leaf)

        # q(r_i|x)
        # tau size=(batch_size, n_topic_root)
        tau = predict_proba_gmm_doc(x, vecs, self.gmm_mu, torch.exp(self.gmm_logvar), torch.exp(self.gmm_pi)) + 1e-9

        # (1)
        # p(x|z)
        log_p_reconst = (torch.mm(x, torch.log(beta)) + torch.mm(1-x, torch.log(1-beta))) / x.shape[1] # vocab_size
        l1 = - torch.mean(torch.sum(gamma * log_p_reconst, axis=1))
        # (2)
        l2 = - torch.mean(torch.sum(tau * torch.mm(gamma, torch.log(alpha)), axis=1)) / len(alpha)
        # (3)
        l3 = - torch.mean(torch.mm(tau, torch.log(torch.exp(self.gmm_pi)).unsqueeze(1)))
        # (4)
        l4 = - torch.mean(torch.sum(gamma * torch.log(gamma), axis=1))
        # (5)
        l5 = - torch.mean(torch.sum(tau * torch.log(tau), axis=1))
        loss = l1 + l2 + l3 + l4 + l5
        # print("loss={:.5f} l1_loss={:.5f} l2_loss={:.5f} l3_loss={:.5f} l4_loss={:.5f} l5_loss={:.5f}".format(loss, l1, l2, l3, l4, l5))

        return loss


        

if __name__ == "__main__":
    model = MyHNTM(
            topic_model_name="gsm",
            encode_dims=[1006, 1024, 512, 300],
            decode_dims=[300, 512, 1006],
            cluster_decode_dims=[1006, 512, 256, 30],
            pretrain_cluster_model_path="../models/pretrain_cluster_model.pkl"
    )
    # for k, v in model.state_dict().items():
    #     print(k) # 参数名称，如encoder.enc_0.weight
    #     print(v.shape) # 参数值，如k.shape = torch.Size([1024, 10000])



