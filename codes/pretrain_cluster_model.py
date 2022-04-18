from matplotlib.pyplot import axis
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from LGY_dataset import MyClusterDataset


class ClusterModel(nn.Module):
	def __init__(self, dims):
		super().__init__()
		self.module = nn.ModuleList([
			nn.Linear(dims[i], dims[i+1])
			for i in range(len(dims)-1)
		])
		self.dropout = nn.Dropout(p=0.01)

	def forward(self, x):
		hid = x
		for i in range(len(self.module)):
			hid = F.relu(self.dropout(self.module[i](hid)))
		return hid


def calc_accuracy(y, y_pred):
	y = y.detach().numpy()
	y_pred = y_pred.detach().numpy()
	y_max = np.argmax(y, axis=1)
	y_pred_max = np.argmax(y_pred, axis=1)
	return np.mean(np.where(y_max==y_pred_max, 1, 0))


def main():
	num_epoch = 30
	batch_size = 128
	lr = 5e-6
	data_source = "netease"
	vocab_size = {"netease": 10171, "20news": 10648}
	n_clusters = 30

	dataset = MyClusterDataset(
		data_source=data_source,
		mode="load"
	)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	model = ClusterModel(dims=[vocab_size[data_source], 512, 256, n_clusters])
	optimizer = torch.optim.SGD(model.parameters(), lr=lr)

	for epoch in range(num_epoch):
		epoch_loss = []
		epoch_acc = []
		for data in dataloader:
			doc, bows, cluster_vec = data
			x = bows
			y = cluster_vec
			y_pred = model(x)
			loss = -1 * torch.sum(y * torch.log_softmax(y_pred, dim=1))
			loss.backward()
			optimizer.step()
			epoch_loss.append(loss.item()/len(bows))
			epoch_acc.append(calc_accuracy(y, y_pred))

		print("Epoch {} AVG Loss: {:.6f}, AVG Acc: {:.6f}".format(
			epoch+1, sum(epoch_loss)/len(epoch_loss), sum(epoch_acc)/len(epoch_acc)))
	
	torch.save(model.state_dict(), "../models/pretrained_cluster_model/{}_pretrained_cluster_model_t{}_v{}.pkl".format(data_source, n_clusters, vocab_size[data_source]))



if __name__ == "__main__":
	main()