from cProfile import run
import torch
from torch.utils.data import DataLoader
import numpy as np
from gensim.models import Word2Vec
from model import *
from dataset import *


class Runner:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.dataloader =  DataLoader(
            self.dataset, batch_size=64, shuffle=True
        )
        self.lr = 1e-3
        self.num_epochs = 50
        self.device = "cpu"

        # 加载预训练词向量
        w2v_path = "/Users/inkding/程序/my-projects/毕设-网易云评论多模态/netease2/models/w2v/c4.mod"
        w2v_model = Word2Vec.load(w2v_path)
        words = list(self.dataset.dictionary.token2id.keys())
        rho_init = []
        for w in words:
            if w2v_model.wv.__contains__(w):
                rho_init.append(w2v_model.wv.get_vector(w))
            else:
                rho_init.append(np.zeros(w2v_model.vector_size))
        rho_init = torch.Tensor(rho_init)

        latent_dim = 300
        topic_size = 30
        vocab_size = self.dataset.vocabsize
        encode_dims = [vocab_size, 512, 256, topic_size]
        self.topic_size = topic_size

        self.model = ClusterDNN(
            encode_dims, latent_dim, topic_size, vocab_size, rho_init)
        
        rho_param_ids = list(map(id, self.model.rho.parameters()))
        base_params = filter(lambda p:id(p) not in rho_param_ids, self.model.parameters())
        self.params = [
            {"params": self.model.rho.parameters(), "lr": 1e-5}, 
            {"params": base_params, "lr":self.lr}
        ]


    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.params, lr=self.lr, momentum=0.9)
        global_loss = [] # 全局纪录
        for epoch in range(self.num_epochs):
            # epoch loss
            epoch_loss = []
            epoch_c_loss = []
            epoch_r_loss = []

            for iter, data in enumerate(self.dataloader):
                optimizer.zero_grad()
                doc, bows = data
                x = bows
                x.to(self.device)
                x_reconst = self.model(x)

                c_loss, r_loss, loss = self.model.loss(x, x_reconst)
                loss.backward()
                optimizer.step()
                epoch_c_loss.append(c_loss.item()/len(bows))
                epoch_r_loss.append(r_loss.item()/len(bows))
                epoch_loss.append(loss.item()/len(bows))

            global_loss.extend(epoch_loss)
            print("Epoch {} AVG c_Loss: {:.6f}".format(epoch+1, sum(epoch_c_loss)/len(epoch_c_loss)))
            print("Epoch {} AVG r_Loss: {:.6f}".format(epoch+1, sum(epoch_r_loss)/len(epoch_r_loss)))
            print("Epoch {} AVG Loss: {:.6f}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
            # print('\n'.join([str(lst) for lst in self.show_topic_words()]))
            # print('='*30)
            if epoch % 10 == 0:
                self.show_topic_words()

        # torch.save(self.model.state_dict(), self.save_path)
        # print("Model saved. path: " + self.save_path)


    def show_topic_words(self, topk=10):
        topic_words = []
        idxes = torch.eye(self.topic_size).to(self.device)
        word_dist = torch.softmax(self.model.decode(idxes), dim=1)
        vals, indices = torch.topk(word_dist, topk, dim=1)
        vals.cpu().tolist()
        indices.cpu().tolist() # 将tensor数据类型转换为基本数据类型
        for i in range(self.topic_size):
            topic_words.append([self.dataset.dictionary.id2token[int(idx)] for idx in indices[i]])
        for words in topic_words:
            print(words)



def main():
    dataset = MyDataset(source_path="./data/docs.txt", mode="load")
    runner = Runner(dataset)
    runner.train()
    # runner.show_topic_words()


if __name__ == "__main__":
    main()