import os
import re
import json
import pickle
from sklearn.cluster import KMeans, SpectralClustering, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np 
from gensim.models import Word2Vec
from data_preprocessing import BertUtils, Word2VecUtils, MyCorpus


class WordEmbeddingCluster:
	def __init__(self, data_source,
				cluster_model_name="kmeans", 
				n_clusters=30, 
				random_state=21):
		'''
		cluster_method: "kmeans", "spectral", "optics"
		'''
		# init data
		self.data_source = data_source
		if data_source == "netease":
			self.vectorizer = Word2VecUtils()
			self.data = open("../data/docs.txt").read()
			self.words = re.split(r' |\n', self.data)
			self.words = list(set(self.words))
			self.data_vectorization() # get self.word2vec
		elif data_source == "20news":
			self.vectorizer = BertUtils()
			with open("../data/20news/20news_small_tfidf_dictionary.pkl", 'rb') as f:
				self.dictionary = pickle.load(f)
				self.words = list(self.dictionary.token2id.keys())
			with open("../data/20news/20news_small_tfidf_vocab_embeds.pkl", 'rb') as f:
				self.vecs = pickle.load(f)
		elif data_source == "wiki103":
			self.vectorizer = BertUtils()
			with open("../data/wiki103/wiki103_small_tfidf_dictionary.pkl", 'rb') as f:
				self.dictionary = pickle.load(f)
				self.words = list(self.dictionary.token2id.keys())
			with open("../data/wiki103/wiki103_small_tfidf_vocab_embeds.pkl", 'rb') as f:
				self.vecs = pickle.load(f)			
		self.results = None
		
		# init cluster model
		self.cluster_model_name = cluster_model_name
		self.n_clusters = n_clusters
		if cluster_model_name == "kmeans":
			self.model = KMeans(n_clusters, random_state=random_state)
		elif cluster_model_name == "gmm":
			self.model = GaussianMixture(n_components=n_clusters, random_state=random_state)

	def data_vectorization(self):
		self.word2vec = {}
		for word in self.words:
			if self.vectorizer.contains_word(word):
				self.word2vec[word] = self.vectorizer.get_word_embedding(word)
		self.vecs = list(self.word2vec.values())
		self.words = list(self.word2vec.keys())

	def fit(self):
		self.model.fit(self.vecs)

	def get_results(self):
		self.results = [[] for i in range(self.n_clusters)]
		if self.cluster_model_name == "kmeans":
			for i, label in enumerate(self.model.labels_):
				self.results[label].append(self.words[i])
		elif self.cluster_model_name == "gmm":
			labels = self.model.predict(self.vecs)
			for i, label in enumerate(labels):
				self.results[label].append(self.words[i])

	def show_results(self):
		if not self.results:
			self.get_results()
		for i, words in enumerate(self.results):
			print("===== label-{} =====".format(i))
			print(words[:50])

	def predict(self, word):
		if not self.results:
			self.get_results()		
		if not self.w2v_model.wv.__contains__(word):
			print("该词不存在")
			return
		vec = self.w2v_model.wv[word]
		if self.cluster_model_name == "kmeans":
			index = self.model.predict([vec])[0]
			print("属于聚类簇-{}，其他词汇如下：".format(index))
			print(self.results[index][:150])

	def save(self):
		self.save_word_cluster_vec()
		self.save_cluster_words()

	def save_word_cluster_vec(self):
		if self.cluster_model_name != "gmm":
			print("失败...只有GMM模型可获取聚类分布向量")
			return
		word2c_vec = {}
		for i in range(len(self.words)):
			word2c_vec[self.words[i]] = self.model.predict_proba([self.vecs[i]])[0]
		with open("../result/{}_word2c_vec_{}_{}.pkl".format(self.data_source, self.cluster_model_name, self.n_clusters), 'wb') as f:
			pickle.dump(word2c_vec, f)

	def save_cluster_words(self):
		if not self.results:
			self.get_results()
		with open("../result/{}_cluster_words_{}.pkl".format(self.data_source, self.n_clusters), 'wb') as f:
			pickle.dump(self.results, f)
			


def main():
	data_source = "20news"
	cluster_model_name = "gmm"
	n_clusters = 30
	cluster_test = WordEmbeddingCluster(data_source, cluster_model_name, n_clusters=n_clusters)
	print("开始训练聚类模型: {}".format(cluster_model_name))
	try:
		cluster_test.fit()
	except KeyboardInterrupt:
		print("键盘中断...程序结束")
	print("聚类簇个数: {}".format(cluster_test.n_clusters))
	cluster_test.show_results()
	cluster_test.save()


if __name__ == '__main__':
	main()




