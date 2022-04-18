from re import L
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from sklearn.mixture import GaussianMixture

'''
本文件包括所有NTM模型。分为VAE和WAE两大类：
    + VAE 类：VAE, NVDM_GSM, AVITM, ETM
    + WAE 类：WAE
拥有类方法：
    + encode
    + decode
    + reparameterize
    + loss
    + forward
'''

# VAE
class VAE(nn.Module):
    def __init__(self, encode_dims, decode_dims=None, dropout=0.01):
        super(VAE, self).__init__()
        # mu and logvar share the first few layers
        self.encoder = nn.ModuleDict({
            "enc_{}".format(i): nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-2)
        })
        self.fc_mu = nn.Linear(encode_dims[-2], encode_dims[-1])
        self.fc_logvar = nn.Linear(encode_dims[-2], encode_dims[-1])
        if decode_dims is not None:
            self.decoder = nn.ModuleDict({
                "dec_{}".format(i): nn.Linear(decode_dims[i], decode_dims[i+1])
                for i in range(len(decode_dims)-1)
            })
        else:
            self.decoder = None
        self.dropout = nn.Dropout(p=dropout)
        

    def encode(self, x):
        # 编码器
        hid = x
        for _, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, logvar = self.fc_mu(hid), self.fc_logvar(hid)
        return mu, logvar

    
    def reparameterize(self, mu, logvar):
        '''
        重参方法，基于encoder给出的期望和对数方差生成z。
        '''
        # torch.randn_like(input): 返回和input大小一致且服从正态分布的tensor
        eps = torch.randn_like(mu)
        std = torch.exp(logvar/2)
        z = mu + eps*std
        return z

    def decode(self, z):
        hid = z
        for i, layer in enumerate(self.decoder.values()):
            hid = layer(hid)
            if i<len(self.decoder)-1:
                hid = F.relu(self.dropout(hid))
        x_reconst = hid
        return x_reconst
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        return x_reconst, mu, logvar



class NVDM_GSM(VAE):
    '''
    正态分布假设。
    '''
    def __init__(self, encode_dims, decode_dims, dropout=0.01):
        super(NVDM_GSM, self).__init__(encode_dims, decode_dims, dropout=0.01)
        self.fc_gsm = nn.Linear(encode_dims[-1], encode_dims[-1])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z_ = self.reparameterize(mu, logvar)
        z = F.softmax(self.fc_gsm(z_), dim=0) # gsm
        x_reconst = self.decode(z)
        return x_reconst, mu, logvar
    
    def loss(self, x, x_reconst, posterior_mean, posterior_logvar):
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失
        kld_loss = -0.5 * torch.sum(1+posterior_logvar-posterior_mean.pow(2)-posterior_logvar.exp()) # 分布约束正则项
        return rec_loss + kld_loss    
            


class AVITM(VAE):
    '''
    狄利克雷分布假设
    '''
    def __init__(self, encode_dims, decode_dims, dropout=0.01, alpha=0.1):
        super(AVITM, self).__init__(encode_dims, decode_dims, dropout=0.01)
        self.num_topics = encode_dims[-1]

        # 设置先验分布参数
        self.prior_mean, self.prior_var = map(nn.Parameter, self.prior(self.num_topics))
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False

    # reparameterize在代码上其实是完全没有变化的

    def prior(self, K, alpha=0.3):
        '''
        Prior for the model.
        :K: number of categories
        :alpha: Hyper param of Dir
        :return: mean and variance tensors
        '''
        # Approximate to normal distribution using Laplace approximation
        a = torch.Tensor(1, K).float().fill_(alpha)
        mean = a.log().t() - a.log().mean(1)
        var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
        return mean.t(), var.t() # Parameters of prior distribution after approximation
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z_ = self.reparameterize(mu, logvar)
        z = F.softmax(z_, dim=0) # avitm在中间使用了softmax
        x_reconst = self.decode(z)
        return x_reconst, mu, logvar   


    def loss(self, x, x_reconst, posterior_mean, posterior_logvar):
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失

        prior_mean = self.prior_mean.expand_as(posterior_mean)
        prior_var = self.prior_var.expand_as(posterior_logvar)
        prior_logvar = self.prior_logvar.expand_as(posterior_logvar)
        var_division = posterior_logvar.exp() / prior_var
        diff = posterior_mean - prior_mean
        diff_term = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        kld_loss = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - self.num_topics)

        return (rec_loss + kld_loss).mean()


class ETM(VAE):
    '''
    正态分布假设
    '''
    def __init__(self, encode_dims, vocab_size, embed_dim, dropout=0.01, rho_init=None):
        '''
        增加参数embed_dim[词嵌入向量大小]，因为ETM在decode过程中需要拆分大小分别为
        (num_topic, embed_dim)及(embed_dim, vocab_size)的两个矩阵。
        VAE中不需要初始化decoder，但是decode方法发生变化。
        '''
        super().__init__(encode_dims, dropout=dropout)
        num_topic = encode_dims[-1]
        self.alpha = nn.Linear(embed_dim, num_topic)
        self.rho = nn.Linear(embed_dim, vocab_size)
        if rho_init is not None:
            self.rho.weight = nn.Parameter(rho_init)
        self.decoder = None # decoder := alpha x rho

    def decode(self, z):
        weight = self.alpha(self.rho.weight) # (vocab_size, num_topic)
        beta = F.softmax(weight, dim=0).transpose(1, 0) # (num_topic, vocab_size)，本质上是decoder
        x_reconst = torch.mm(z, beta)
        return x_reconst

    def loss(self, x, x_reconst, posterior_mean, posterior_logvar):
        # 和NVDM_GSM是一样的
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失
        kld_loss = -0.5 * torch.sum(1+posterior_logvar-posterior_mean.pow(2)-posterior_logvar.exp()) # 分布约束正则项
        return rec_loss + kld_loss    
            
        


class WAE(nn.Module):
    def __init__(self, encode_dims, decode_dims=None, dropout=0.01):
        super().__init__()
        self.encoder = nn.ModuleDict({
            "enc_{}".format(i): nn.Linear(encode_dims[i], encode_dims[i+1])
            for i in range(len(encode_dims)-1)
        })
        if decode_dims is not None:
            self.decoder = nn.ModuleDict({
                "dec_{}".format(i): nn.Linear(decode_dims[i], decode_dims[i+1])
                for i in range(len(decode_dims)-1)
            })
        else:
            self.decoder = None            
        self.num_topic = encode_dims[-1]
        self.dropout = nn.Dropout(p=dropout)
    

    def encode(self, x):
        hid = x
        for i, (_,layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder)-1: # 隐变量层前不经过非线性变换
                hid = F.relu(hid)
        return hid

    def decode(self, z):
        hid = z
        for i, (_,layer) in enumerate(self.decoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.decoder)-1: # 输出层前不经过非线性变换
                hid = F.relu(hid)
        return hid
    

    def forward(self, x):
        z = self.encode(x)
        theta = F.softmax(z, dim=1)
        x_reconst = self.decode(theta)
        return x_reconst, theta

    
    def sample(self, dist="dirichlet", batch_size=128, dirichlet_alpha=0.1, ori_data=None):
        if dist=="dirichlet":
            z_true = np.random.dirichlet(
                np.ones(self.num_topic)*dirichlet_alpha, size=batch_size
            )
            z_true = torch.from_numpy(z_true).float()
            return z_true
        elif dist=="gaussian":
            z_true = np.random.randn(batch_size, self.num_topic)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true
    

    def loss(self, x, x_reconst, theta_prior, theta_post, device, beta):
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失
        mmd_loss = self.mmd_loss(theta_prior, theta_post, device)
        return rec_loss + mmd_loss*beta


    def mmd_loss(self, x, y, device, t=0.1, kernel="diffusion"):
        '''
        使用information diffusion kernel计算mmd_loss。
        '''
        eps = 1e-6
        n, d = x.shape # (batch_size, num_topic)
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(i+1, n):
                    sum_xx = sum_xx + torch.norm(x[i]-x[j], p=1).to(device)
            sum_xx = sum_xx / (n * (n-1))

            sum_yy = torch.zeros(1).to(device)
            for i in range(y.shape[0]):
                for j in range(i+1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0]-1))

            sum_xy = torch.zeros(1).to(device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i]-y[j], p=1).to(device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1-eps), t, d-1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1-eps), t, d-1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1-eps), t, d-1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n-1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n-1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy



class WETM(WAE):
    def __init__(self, encode_dims, embed_dim, vocab_size, rho_init=None, dropout=0.01):
        '''
        继承WAE的编码方式和损失函数，将解码拆分为两个矩阵的乘机（同ETM）
        '''
        super().__init__(encode_dims, dropout=dropout)
        num_topic = encode_dims[-1]
        self.alpha = nn.Linear(embed_dim, num_topic)
        self.rho = nn.Linear(embed_dim, vocab_size)
        if rho_init is not None:
            self.rho.weight = nn.Parameter(rho_init) # 只有nn.Parameter类型的参数才会参与训练
        self.decoder = None # decoder := alpha x rho

    
    def decode(self, z):
        weight = self.alpha(self.rho.weight)
        beta = F.softmax(weight, dim=0).transpose(1, 0)
        x_reconst = torch.mm(z, beta)
        return x_reconst


class VaDE(VAE):
    def __init__(self, encode_dims, decode_dims=None, n_clusters=10, dropout=0.01):
        super().__init__(encode_dims, decode_dims=decode_dims, dropout=dropout)
        # 保留 self.encoder, decoder, fc_mu, fc_logvar, dropout
        self.n_clusters = n_clusters
        self.latent_dim = encode_dims[-1]
        self.fc_pi = nn.Linear(encode_dims[-2], n_clusters)
        self.fc1 = nn.Linear(encode_dims[-1], encode_dims[-1])

        self.pi = nn.Parameter(torch.ones(n_clusters, dtype=torch.float32) / n_clusters,
                                requires_grad=True)
        self.mu_c = nn.Parameter(torch.zeros(n_clusters, self.latent_dim, dtype=torch.float32), requires_grad=True)
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, self.latent_dim, dtype=torch.float32), requires_grad=True)

        self.gmm = GaussianMixture(n_components=n_clusters,
                                    covariance_type="diag", max_iter=200, reg_covar=1e-5)

    def encode(self, x):
        hid = x
        for _, layer in self.encoder.items():
            hid = F.relu(self.dropout(layer(hid)))
        mu, logvar, qc = self.fc_mu(hid), self.fc_logvar(hid), F.softmax(self.fc_pi(hid), dim=1)
        return mu, logvar, qc

    # decode保持不变
    def forward(self, x, collate_fn=None, isPretrain=False):
        mu, logvar, qc = self.encode(x)
        if isPretrain==False:
            _z = self.reparameterize(mu, logvar)
        else:
            _z = mu
        _theta = self.fc1(_z)   #TBD
        if collate_fn!=None:
            theta = collate_fn(_theta)
        else:
            theta = _theta
        x_reconst = self.decode(theta)
        return x_reconst, mu, logvar

    def get_latent(self,x):
        with torch.no_grad():
            mu, logvar, qc = self.encode(x)
            return mu
    
    def loss(self, x, x_reconst, mu_posterior, logvar_posterior, beta=1.0, gamma=1e-7):
        # 由 rec_loss, kl_loss, mus_mutual_distance 三部分组成
        logsoftmax= torch.log_softmax(x_reconst, dim=1)
        rec_loss = -1.0 * torch.sum(x*logsoftmax) # 重构损失
        kl_loss = self.compute_kl_loss(mu_posterior, logvar_posterior)        
        mus_mutual_distance = self.compute_mus_mutual_distance()
        print("rec_loss", rec_loss)
        print("kl_loss", kl_loss)
        print("mus_mutual_distance", mus_mutual_distance)

        return rec_loss + kl_loss*beta + mus_mutual_distance*gamma

    def compute_kl_loss(self, mus, logvars):
        # mus=[batch_size,latent_dim], logvars=[batch_size,latent_dim]
        zs = self.reparameterize(mus,logvars)
        # zs=[batch_size,latent_dim]
        mu_c = self.mu_c
        logvar_c = self.logvar_c
        # mu_c=[n_clusters,latent_dim], logvar_c=[n_clusters,latent_dim]
        delta = 1e-10
        gamma_c = torch.exp(torch.log(self.pi.unsqueeze(0))+self.log_pdfs_gauss(zs,mu_c,logvar_c))+delta
        gamma_c = gamma_c / (gamma_c.sum(dim=1).view(-1,1))
        #gamma_c = F.softmax(gamma_c*len(gamma_c)*1.2,dim=1) #amplify the discrepancy of the distribution
        # gamma_c=[batch_size,n_clusters]

        # kl_div=[batch_size,n_clusters,latent_dim], the 3 lines above are 
        # correspond to the 3 terms in the second line of Eq. (12) in the original paper, respectively.
        kl_div = 0.5 * torch.mean(torch.sum(gamma_c*torch.sum(logvar_c.unsqueeze(0)+
                        torch.exp(logvars.unsqueeze(1)-logvar_c.unsqueeze(0))+
                        (mus.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/(torch.exp(logvar_c.unsqueeze(0))),dim=2),dim=1))
        # The two sum ops are corrrespond to the Sigma wrt J and the Sigma wrt K, respectively.
        # torch.mean() is applied along the batch dimension.
        kl_div -= torch.mean(torch.sum(gamma_c*torch.log(self.pi.unsqueeze(0)/gamma_c),dim=1)) + 0.5*torch.mean(torch.sum(1+logvars,dim=1))
        # Correspond to the last two terms of Eq. (12) in the original paper.    
        return kl_div    

    
    def compute_mus_mutual_distance(self, dist_type='cosine'):
        if dist_type=='cosine':
            norm_mu = self.mu_c/torch.norm(self.mu_c, dim=1, keepdim=True)
            cos_mu = torch.matmul(norm_mu, norm_mu.transpose(1,0))
            cos_sum_mu = torch.sum(cos_mu) # the smaller the better
        
            theta = F.softmax(self.fc1(self.mu_c),dim=1)
            cos_theta = torch.matmul(theta, theta.transpose(1,0))
            cos_sum_theta = torch.sum(cos_theta)
        
            dist = cos_sum_mu + cos_sum_theta
        else:
            mu = self.mu_c
            dist = torch.reshape(torch.sum(mu**2,dim=1),(mu.shape[0],1))+ torch.sum(mu**2,dim=1)-2*torch.matmul(mu,mu.t())
            dist = 1.0/(dist.sum() * 0.5) + 1e-12 #/ (len(mu)*(len(mu)-1)), use its inverse, then the smaller the better
        return dist    


    def log_pdfs_gauss(self, z, mus, logvars):
        # Compute log value of the posterion probability of z given mus and logvars under GMM hypothesis.
        # i.e. log(p(z|c)) in the Equation (16) (the second term) of the original paper.
        # params: z=[batch_size * latent_dim], mus=[n_clusters * latent_dim], logvars=[n_clusters * latent_dim]
        # return: [batch_size * n_clusters], each row is [log(p(z|c1)),log(p(z|c2)),...,log(p(z|cK))]
        log_pdfs = []
        for c in range(self.n_clusters):
            log_pdfs.append(self.log_pdf_gauss(z,mus[c:c+1,:],logvars[c:c+1,:]))
        return torch.cat(log_pdfs,dim=1)    


    def log_pdf_gauss(self, z ,mu, logvar):
        # Compute the log value of the probability of z given mu and logvar under gaussian distribution
        # i.e. log(p(z|c)) in the Equation (16) (the numerator of the last term) of the original paper
        # params: z=[batch_size * latent_dim], mu=[1 * latent_dim], logvar=[1 * latent_dim]
        # return: res=[batch_size,1], each row is the log val of the probability of a data point w.r.t the component N(mu,var)
        '''
            log p(z|c_k) &= -(J/2)log(2*pi) - (1/2)*\Sigma_{j=1}^{J} log sigma_{j}^2 - \Sigma_{j=1}^{J}\frac{(z_{j}-mu_{j})^2}{2*\sigma_{j}^{2}}
                         &=-(1/2) * \{[log2\pi,log2\pi,...,log2\pi]_{J}
                                    + [log\sigma_{1}^{2},log\sigma_{2}^{2},...,log\sigma_{J}^{2}]_{J}
                                    + [(z_{1}-mu_{1})^2/(sigma_{1}^{2}),(z_{2}-mu_{2})^2/(sigma_{2}^{2}),...,(z_{J}-mu_{J})^2/(sigma_{J}^{2})]_{J}                   
                                    \},
            where J = latent_dim
        '''
        return (-0.5*(torch.sum(np.log(2*np.pi)+logvar+(z-mu).pow(2)/torch.exp(logvar),dim=1))).view(-1,1)



    
if __name__=="__main__":
    model = WETM([10000, 1024, 512, 20], 300, 1000)
    print(len(model.parameters()))