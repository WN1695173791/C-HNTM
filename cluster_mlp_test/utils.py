import numpy as np
import torch


def calc_batch_cluster_loss(x, vocab_matrix, cluster_size=20):
    total_loss = []
    for i in range(x.shape[0]):
        total_loss.append(calc_cluster_loss(x[i], vocab_matrix))
    return torch.mean(torch.Tensor(total_loss))


def calc_batch_accept_k_reconst_loss(x, x_reconst, accept_k=3):
    total_loss = []
    for i in range(x.shape[0]):
        total_loss.append(calc_accept_k_reconst_loss(x[i], x_reconst[i]))
    return torch.mean(torch.Tensor(total_loss))



def calc_cluster_loss(x, vocab_matrix, cluster_size=20):
    '''
    param: x: torch.Tensor, softmax分布
    param: vocab_matrix: numpy.ndarray, 词向量矩阵
    '''
    new_ids = np.argsort(x)[::-1][:cluster_size]
    vocab_matrix = np.array(vocab_matrix)
    new_x = standardization(x[new_ids])
    new_matrix = vocab_matrix[new_ids]
    similar_matrix = get_cos_similar_matrix(new_matrix, new_matrix)
    loss = 0
    for i in range(len(new_ids)):
        for j in range(i+1, len(new_ids)):
            loss += new_x[i] * new_x[j] / similar_matrix[i][j]
    return loss * 2


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def calc_accept_k_reconst_loss(x, x_reconst, accept_k=10):
    ids_topk = np.argsort(x_reconst)[::-1][:accept_k]
    new_x_reconst = standardization(x_reconst[ids_topk])
    loss = 0
    for i in range(len(ids_topk)):
        id = ids_topk[i]
        if x[id]>0:
            loss += new_x_reconst[i]
    return -loss*1000



def standardization(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x = (x-mu) / sigma
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range    