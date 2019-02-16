import pandas as pd 
import numpy as np
import sys
import matplotlib.pyplot as plt

class q1(object):
	def __init__(self):
		self.read_data = pd.read_csv("./carbig.csv", header=None)
		self.data = np.split(self.read_data, [6], axis=1)

		self.X = self.data[0].values
		self.y = self.data[1].values

		self.X_horse_power = self.X[:, [2]]
		self.X_model_year = self.X[:, [5]]

		self.degrees = [0, 1, 2, 3, 4, 5]
		self.colors = ["r", "g", "k", "y", "c", "m"]

		self.X_train, self.X_test, self.y_train, self.y_test = [], [], [], []
		self.weights = []

	def divide(self, X, y):
		self.X_train, self.X_test, self.y_train, self.y_test = X[:300, :], X[300:, :], y[:300, :], y[300:, :]

	def calculate_mean_error(self, X, w, y, N):
		e = np.dot(X,w)
		return float(np.sum((y - e)**2)) / float(N)

	def calculate_rank(self, X):
		return np.linalg.matrix_rank(X)

	def train(self):
		a = np.linalg.inv(np.dot(self.X_train.T, self.X_train))
		b = np.dot(self.X_train.T, self.y_train)
		self.weights = np.dot(a, b)

	def change_X(self, X):
		for degree in self.degrees[2:]:
			new_col = X[:, [0]] ** degree
			X = np.concatenate([X, new_col], axis = 1)
		return X

	def centralize(self, X):
		mean = np.mean(X, axis = 0)
		std = np.mean(X, axis = 0)
		X = (X - mean) / std
		return X
 
	def plot_regression_line(self, X, w, d):
		X = np.sort(X, axis = 0)
		y_pred = np.dot(X, w)
		X_org = np.sort(self.X_horse_power[:300:, [1]], axis = 0)
		plt.plot(X_org, y_pred, self.colors[d], label = "p = {}".format(d), linewidth = 2, zorder = 20) 

	def q1_2(self):
		rank = self.calculate_rank(np.dot(self.X.T, self.X))
		print "Rank of XT.X: {}".format(rank)

	def q1_3(self):
		self.train()
		training_mean = self.calculate_mean_error(self.X_train, self.weights, self.y_train, len(self.y_train))
		test_mean = self.calculate_mean_error(self.X_test, self.weights, self.y_test, len(self.y_test))
		
		print "Model Coffecient values: "
		print self.weights
		print "Mean squared error on training set: {}".format(training_mean)
		print "Mean squared error in test set: {}".format(test_mean)

	def q1_5(self):
		plt.scatter(self.X_horse_power, self.y)
		plt.xlabel('Horse power')
		plt.ylabel('MPG')
		plt.title("q1_5")
		plt.show()

	def q1_6(self):
		self.X_horse_power = self.change_X(self.X_horse_power)
		ones = np.ones((self.X_horse_power.shape[0], 1))
		self.X_horse_power = np.concatenate([ones, self.X_horse_power], axis = 1)	

		rank_dic = {}
		p = []
		for degree in self.degrees:
			p.append(degree)
			rank_dic[degree] = self.calculate_rank(np.dot(self.X_horse_power[:, p].T, self.X_horse_power[:, p]))
		print "Ranks before centralizing"
		print rank_dic

		### Centralizing the data ###		
		self.X_horse_power = self.centralize(self.X_horse_power)
		self.X_horse_power[:, 0] = 1
		rank_dic = {}
		p = []
		for degree in self.degrees:
			p.append(degree)
			rank_dic[degree] = self.calculate_rank(np.dot(self.X_horse_power[:, p].T, self.X_horse_power[:, p]))
		print "Ranks after centralizing"
		print rank_dic
		
	def q1_7(self):
		print "Values for 1.7"
		p = []
		plt.scatter(self.X_horse_power[:300:, [1]], self.y[:300, :], zorder = 1, label = 'Training data')
		plt.xlabel('Horse power')
		plt.ylabel('MPG')
		plt.title("q1_7")
		for degree in self.degrees:
			p.append(degree)
			self.divide(self.X_horse_power[:, p], self.y)
			self.train()
			self.plot_regression_line(self.X_train, self.weights, degree)
			training_mean = self.calculate_mean_error(self.X_train, self.weights, self.y_train, len(self.y_train))
			test_mean = self.calculate_mean_error(self.X_test, self.weights, self.y_test, len(self.y_test))

			print "Training and Testing for p = {}".format(degree)
			print "Mean squared error on training set: {}".format(training_mean)
			print "Mean squared error in test set: {}".format(test_mean)
		plt.legend() 
		plt.show()
		
	def q1_8(self):
		print "Values for 1.8"
		p = [0]
		b = []
		self.X_model_year = self.centralize(self.X_model_year)
		self.X_model_year = self.change_X(self.X_model_year)

		self.degrees = [1, 2, 3]
		for degree in self.degrees:
			p.append(degree)
			b.append(degree)
			X_joint = np.concatenate([self.X_horse_power[:, p], self.X_model_year[:, b]], axis = 1)
			self.divide(X_joint, self.y)
			self.train()
			training_mean = self.calculate_mean_error(self.X_train, self.weights, self.y_train, len(self.y_train))
			test_mean = self.calculate_mean_error(self.X_test, self.weights, self.y_test, len(self.y_test))

			print "Training and Testing for p = {}".format(degree)
			print "Mean squared error on training set: {}".format(training_mean)
			print "Mean squared error in test set: {}".format(test_mean)


q1 = q1()

q1.divide(q1.X, q1.y)
q1.q1_2()
q1.q1_3()
q1.q1_5()
q1.q1_6()
q1.q1_7()
q1.q1_8()


