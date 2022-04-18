import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class ClusterDNN(nn.Module):
    def __init__(self, encode_dims, latent_dim, topic_size, vocab_size, rho_init=None, dropout=0.01):
        super().__init__()
        assert encode_dims[0]==vocab_size, "encode_dims[0]({})不等于vocab_size({})".format(encode_dims[0], vocab_size)
        assert encode_dims[-1]==topic_size, "encode_dims[-1]({})不等于topic_size({})".format(encode_dims[-1], topic_size)
        self.rho_init = rho_init
        self.encoder = nn.ModuleList([
            nn.Linear(encode_dims[i], encode_dims[i+1]) for i in range(len(encode_dims)-1)
        ])
        self.alpha = nn.Linear(latent_dim, topic_size)
        self.rho = nn.Linear(latent_dim, vocab_size)
        if rho_init is not None:
            self.rho.weight = nn.Parameter(rho_init)
        self.dropout = nn.Dropout(p=dropout)
        
    
    def encode(self, x):
        hid = x
        for i in range(len(self.encoder)):
            hid = F.relu(self.dropout(self.encoder[i](hid)))
        return hid

    def decode(self, hid):
        weight = self.alpha(self.rho.weight)
        beta = F.softmax(weight, dim=0).transpose(1, 0)
        x_reconst = torch.mm(hid, beta)
        return x_reconst

    def forward(self, x):
        hid = self.encode(x)
        x_reconst = self.decode(hid)
        return x_reconst
    

    def loss(self, x, x_reconst, a=1):
        cluster_loss = calc_batch_cluster_loss(torch.softmax(x_reconst, dim=1).detach().numpy(), self.rho_init)
        reconst_loss = calc_batch_accept_k_reconst_loss(x, x_reconst.detach().numpy())
        total_loss = a*cluster_loss + (1-a)*reconst_loss
        total_loss.requires_grad_()
        # print("cluster_loss: {:.3f}, reconst_loss: {:.3f}, total_loss: {:.3f}".format(cluster_loss, reconst_loss, total_loss))
        return cluster_loss, reconst_loss, total_loss



