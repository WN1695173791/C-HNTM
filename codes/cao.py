from statistics import mean
import numpy as np
import torch
from utils import predict_proba_gmm_doc
from dataset import MyDataset

def main():
    means = torch.rand((20, 768))
    covariances = torch.rand((20, 768))
    weights = torch.rand(20)
    

    dataset = MyDataset("20news")
    vecs = torch.tensor(np.array(dataset.vecs))
    doc_bow = []
    for i in range(30):
        doc, bow = dataset[i]
        doc_bow.append(bow)
    doc_bow = torch.stack(doc_bow)
    print(doc_bow.shape)
    print(predict_proba_gmm_doc(doc_bow, vecs, means, covariances, weights))

if __name__ == "__main__":
    main()
