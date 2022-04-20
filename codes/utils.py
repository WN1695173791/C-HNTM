import os
import torch
import pickle
import numpy as np
from gensim.models.coherencemodel import CoherenceModel


def get_device(device_id=0):
    if device_id==-1 or not torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cuda:{}".format(device_id))

def get_or_create_path(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return path


def calc_topic_diversity(topic_words):
    '''
    计算话题多样性。
    param: topics_words[List[List[str]]]: [[topic1_w1, topic1_w2, ...], [topic2_w1, ...], ...]
    return[float]: topic_diversity
    '''
    vocab = set()
    for words in topic_words:
        vocab.update(set(words))
    num_words_total = len(topic_words)*len(topic_words[0])
    topic_div = len(vocab)/num_words_total
    return topic_div


def calc_intra_topic_similarity(topic_words_list, w2v_model, mode="avg"):
    '''
    计算话题内部的词相似度。
    '''
    res = 0
    for topic_words in topic_words_list:
        valid_words = []
        for w in topic_words:
            if w2v_model.wv.__contains__(w):
                valid_words.append(w)
        simi_mtx = []
        for i in range(len(valid_words)):
            simi_vec = list(map(lambda w: w2v_model.wv.similarity(valid_words[i], w), valid_words))
            simi_vec[i] = 0
            simi_mtx.append(simi_vec)
        if mode=="avg":
            score_vec = np.mean(np.array(simi_mtx), axis=1)
        elif mode=="max":
            score_vec = np.max(np.array(simi_mtx), axis=1)
        res += np.mean(score_vec)
    return np.mean(res)
    
    


def calc_topic_coherence(topic_words, docs, dictionary):
    '''
    计算话题凝聚程度，包括 c_v, c_w2v, c_uci, c_npmi
    params:
        topics_words[List[List[str]]]: [[topic1_w1, topic1_w2, ...], [topic2_w1, ...], ...]
        docs[List[List[str]]]: [[doc1_w1, doc1_w2, ...], [doc2_w1, ...], ...]
        dictionary[gensim.corpora.Dictionary]
    return[dict]:
        {"cv": [cv_score, cv_score_per_topic], "cw2v": ...}
    '''
    # cv_score
    cv_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence="c_v", processes=1)
    cv_score = cv_model.get_coherence()

    # # cw2v_score

    # cuci_score
    cuci_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence="c_uci", processes=1)
    cuci_score = cuci_model.get_coherence()

    # cnpmi_score
    cnpmi_model = CoherenceModel(topics=topic_words, texts=docs, dictionary=dictionary, coherence="c_npmi", processes=1)
    cnpmi_score = cnpmi_model.get_coherence()

    score_dict = {
        "metric/cv": cv_score,
        "metric/cuci": cuci_score,
        "metric/cnpmi": cnpmi_score
    }

    return score_dict


def random_cluster_vec_init(n_clusters=30):
    with open("../models/corpus/dictionary.pkl", 'rb') as f:
        dictionary = pickle.load(f)
    words = list(dictionary.token2id.keys())
    vocab_size = len(words)

    cluster_vec_mtx = np.random.random((vocab_size, n_clusters))
    cluster_vec_mtx = np_softmax(cluster_vec_mtx)    

    word2c_vec = {}
    for i in range(len(words)):
        word2c_vec[words[i]] = cluster_vec_mtx[i]
    with open("../result/word2c_vec.pkl", "wb") as f:
        pickle.dump(word2c_vec, f)
    


def np_softmax(x):
    e = np.exp(x)
    softmax = e / np.sum(e, axis=1).reshape(-1, 1)
    return softmax


def _estimate_log_gaussian_prob(X, means, precisions_chol):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = torch.sum(torch.log(precisions_chol), axis=1)
    # print("log_det.shape", log_det.shape, log_det[0])
    precisions = precisions_chol ** 2
    # print("precisions.shape", precisions.shape, precisions[0])
    log_prob = (torch.sum((means**2 * precisions), 1) -
                2.*torch.mm(X, (means * precisions).T) +
                torch.mm(X**2, precisions.T))
    # print("log_prob.shape", log_prob.shape, log_prob[0])
    return -0.5 * (n_features * np.log(2*np.pi) + log_prob) + log_det


def _estimate_weighted_log_prob(X, means, precisions_chol, weights):
    return _estimate_log_gaussian_prob(X, means, precisions_chol) + torch.log(weights)

def _estimate_log_prob_resp(X, means, precisions_chol, weights):
    weighted_log_prob = _estimate_weighted_log_prob(X, means, precisions_chol, weights)
    log_prob_norm = torch.logsumexp(weighted_log_prob, axis=1)
    # print("weighted_log_prob.shape", weighted_log_prob.shape, weighted_log_prob[0])
    # print("log_prob_norm.shape", log_prob_norm.shape, log_prob_norm[0])
    # with np.errstate(under='ignore'):
    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    # print("log_resp.shape", log_resp.shape, log_resp[0])
    return log_prob_norm, log_resp


def _predict_proba_gmm(X, means, precision_chol, weights):
    '''
    计算高斯混合模型下X的概率分布
    默认covariance_type=='diag'
    '''
    _, log_resp = _estimate_log_prob_resp(X, means, precision_chol, weights)
    return torch.exp(log_resp)



def predict_proba_gmm_doc(doc_X, vecs, means, covariances, weights):
    '''
    获得p(r|x)，即在文档的条件下顶层主题的概率，等于文档中所有词的条件下顶层主题概率取平均
    params:
        doc_X: 文档词袋表示，size=(batch_size, vocab_size)
        vecs: 词向量集，索引与词袋一致，size=(vocab_size, embed_dim)
        means: gmm模型的means
        covariances: gmm模型的covariances
        weights: gmm模型的weights
    return: 概率矩阵，size=(batch_size, n_topic_root)
    '''
    precision_chol = 1. / torch.sqrt(covariances)
    res = []
    for i in range(len(doc_X)): # 沿batch_size方向遍历
        word_vecs = vecs[torch.where(doc_X[i]>0)]
        proba_gmm_doc = torch.mean(_predict_proba_gmm(word_vecs, means, precision_chol, weights), axis=0)
        res.append(proba_gmm_doc)
    res = torch.stack(res)
    return res



if __name__ == "__main__":
    doc_X = torch.rand((128, 10256))
    doc_X = torch.where(doc_X>0.8, 1, 0)
    vecs = torch.rand((10256, 768))
    means = torch.rand((20, 768))
    covariances = torch.rand((20, 768))
    weights = torch.rand(20)

    print(predict_proba_gmm_doc(doc_X, vecs, means, covariances, weights))