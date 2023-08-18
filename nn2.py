from google.colab import drive
drive.mount('/content/drive')



from audioop import bias
import sys
import os
import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90
class Net(object):
	'''
	'''
	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
		self.num_layers = num_layers
		self.num_units = num_units

		self.biases = []
		self.weights = []
		for i in range(num_layers):

			if i==0:
				# Input layer
				self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
			else:
				# Hidden layer
				self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

			self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

	def __call__(self, X):
		'''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
		self.net = []
		self.net.append(X)
		self.O = []
		for i, (w, b) in enumerate(zip(self.weights, self.biases)):
			self.O.append(X)
			net_i = np.dot(X, w) + b.T
			self.net.append(net_i)
			if i == len(self.weights):
				X = net_i
				break
			X = relu(net_i)
		self.pred = X
		return self.pred

	def backward(self, X, y, lamda):
		'''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
		delta_weights = []
		delta_biases = []

		back_outs = []
		back_act_outs = []

		loss = (self.pred - y)
		batch_size = y.shape[0]
		cur_layer = self.num_layers - 1

		back_outs.insert(0, loss)
		back_act_outs.insert(0, loss)
		delta_weights.insert(0, (np.dot(self.O[cur_layer+1].T, back_outs[0])/batch_size)+lamda * self.weights[cur_layer+1])
		delta_biases.insert(0, (np.array(np.sum(back_outs[0], axis=0).T))/batch_size )
		delta_biases[0] = delta_biases[0].reshape(delta_biases[0].shape[0], 1)
		delta_biases[0] = delta_biases[0] + lamda * self.biases[cur_layer+1]

		for i in range(1, self.num_layers + 1):
			back_act_outs.insert(0, np.dot(back_outs[0], self.weights[cur_layer + 1].T))
			back_outs.insert(0, back_act_outs[0] * d_relu(self.O[cur_layer + 1]))
			delta_weights.insert(0, (np.dot(self.O[cur_layer].T, back_outs[0])/batch_size)+lamda * self.weights[cur_layer])
			delta_biases.insert(0, (np.array(np.sum(back_outs[0], axis=0).T))/batch_size )
			delta_biases[0] = delta_biases[0].reshape(delta_biases[0].shape[0], 1)
			delta_biases[0] = delta_biases[0] + lamda * self.biases[cur_layer]
			cur_layer = cur_layer - 1
		
		return delta_weights, delta_biases

def relu(X):
	a = np.maximum(X,0)
	return a

def d_relu(X):
	X[X <= 0] = 0
	X[X > 0] = 1
	return X

class Optimizer(object):
	'''
	'''

	def __init__(self, learning_rate):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		self.learning_rate = learning_rate

	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		'''

		for index in range(len(weights)):
			weights[index] = weights[index] - self.learning_rate * delta_weights[index]
			biases[index] = biases[index] - self.learning_rate * delta_biases[index]
		
		return weights, biases


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	'''
	return np.sum((y-y_hat)**2)/len(y)

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	loss = 0
	for i in range(len(weights)):
		loss = loss + np.sum(np.square(weights[i])) + np.sum(np.square(biases[i]))
	return loss

def loss_fn(y, y_hat, weights, biases, lamda):
	'''
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	return loss_mse(y,y_hat) + (lamda * loss_regularization(weights, biases))

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	#root[sum[(y-y_hat)^2/n]
	n = len(y)
	return np.sqrt((np.sum(y-y_hat)**2)/n)








def read_data():
	'''
	Read the train, dev, and test datasets
	'''
	train_input = pd.read_csv('/content/drive/MyDrive/Manoj/Manoj/regression/data/train.csv').iloc[:, 1:].to_numpy()
	dev_input = pd.read_csv('/content/drive/MyDrive/Manoj/Manoj/regression/data/dev.csv').iloc[:, 1:].to_numpy()
	test_input = pd.read_csv('/content/drive/MyDrive/Manoj/Manoj/regression/data/test.csv').to_numpy()

	train_target = pd.read_csv('/content/drive/MyDrive/Manoj/Manoj/regression/data/train.csv').iloc[:, 0].to_numpy()
	dev_target = pd.read_csv('/content/drive/MyDrive/Manoj/Manoj/regression/data/dev.csv').iloc[:, 0].to_numpy()
	train_target = train_target.reshape(train_target.shape[0], 1)
	dev_target = dev_target.reshape(dev_target.shape[0], 1)
	return train_input, train_target, dev_input, dev_target, test_input










def train(net, optimizer, lamda, batch_size, max_epochs,train_input, train_target,dev_input, dev_target, test_input):

	m = train_input.shape[0]
	batchloss = []
	epoch = []  

	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss
		epoch.append(e)
		batchloss.append(batch_loss)
   
		print('Epoch:',e,'Rmse loss:',rmse(batch_target, pred),'Batch Loss:', batch_loss)
	dev_pred = net(dev_input)
	test_preds = net(test_input)
	dev_rmse = rmse(dev_target, dev_pred)
	print('RMSE on dev data: {:.5f}'.format(dev_rmse))
	return batchloss, epoch, test_preds

 
 




def main():

	# Hyper-parameters 
	max_epochs = 100
	batch_size = 64
	learning_rate = 1e-4
	num_layers = 7
	num_units = 5
	lamda = 0.12 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units)
	optimizer = Optimizer(learning_rate)
	bt_loss, epoch, test_pred = train(
		net, optimizer, lamda, batch_size, max_epochs,dev_input, dev_target,
		train_input, train_target, test_input
	)
	return net, bt_loss, epoch, test_preds


if __name__ == '__main__':
	net, bt_loss, epoch, test_preds = main()
    
    
    
import matplotlib.pyplot as plt

plt.plot(epoch, bt_loss)
plt.xlabel('Epochs')
plt.ylabel('Batchloss')
plt.title("Loss vs Epoch for batch size = 64")