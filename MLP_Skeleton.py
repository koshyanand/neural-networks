"""
Anand P Koshy
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
from sklearn.preprocessing import normalize, scale
import math
import matplotlib.pyplot as plt
# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	# DEFINE __init function
        self.W = W
        self.b = b

    def forward(self, x):
	# DEFINE forward function
        # print(type(self.W))
        # print(type(self.W))
        # print(self.W)
        # print(x.shape)
        # print(self.W.shape)
        self.x = x
        # print("X : ", x.shape)

        op = np.matmul(self.W, x.T) + self.b
        op = op.T
        # print("OP : ", op.shape)
        return op

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        # print("LT : X : ", self.x.shape)
        return np.dot(grad_output.T, self.x), grad_output
	# DEFINE backward function
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function
        self.relu_op = np.maximum(0, x)
        # print("Relu Forward Shape : ", self.relu_op.shape)
        return self.relu_op

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        dz = np.zeros((grad_output.shape))
        dz[self.relu_op<0] = 0
        dz[self.relu_op>0] = 1
        dz[self.relu_op == 0] = np.random.uniform(0.01,1)

        # print("ReluOP : ", self.relu_op.shape)
        return grad_output * self.relu_op

    # DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object) :
    def forward(self, x):
		# DEFINE forward function

        round_x = np.round(x, 4)
        self.yhat = 1.0 / (1.0 + np.exp(np.negative(round_x)))
        # print(self.yhat.shape)
        # out_array = np.clip(self.yhat, a_min = e , a_max = 6) 

        # print("yhat : ", self.yhat[0:30])
        return self.yhat

    def backward(
	    self,
	    grad_output, 
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):
		# DEFINE backward function
        # ADD other operations and data entries in SigmoidCrossEntropy if needed

        # delta =  (-1 / len(self.yhat)) * (grad_output / self.yhat - (1 - grad_output) / (1 - self.yhat)) * (self.yhat * (1 - self.yhat))
        delta =  (self.yhat - grad_output)        
        return delta

# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network
        self.W1 = np.round(np.random.rand(hidden_units, input_dims), 5) * 0.1
        self.W2 = np.round(np.random.rand(1, hidden_units), 5) * 0.1
        # print(np.amax(self.W1), " : ", np.amin(self.W1))
        # print(np.amax(self.W2), " : ", np.amin(self.W2))
        self.B1 = np.full((hidden_units, 1), 1)
        self.B2 = np.full((1, 1), 1)

        self.mdW1 = np.zeros((self.W1.shape))
        self.mdW2 = np.zeros((self.W2.shape))

        self.mdB1 = np.zeros((self.B1.shape))
        self.mdB2 = np.zeros((self.B2.shape))

    def calculate_loss(self, yhat, y_batch, l2_penalty):
        L2_reg_cost = (l2_penalty / (2 * y_batch.shape[0])) * (np.sum((self.W1)**2) + np.sum((self.W2)**2))
        pos_log = np.log(yhat + 1e-15)
        neg_log = np.log(1 - yhat + 1e-15)
        loss =  (np.multiply(y_batch, pos_log) + np.multiply((1 - y_batch), neg_log)) 
        loss = -1*np.mean(loss, axis = 0) 
        loss += L2_reg_cost
        return loss

    def predict(self, x, y):
        lt = LinearTransform(self.W1, self.B1)
        a1 = lt.forward(x)
        a1 = scale( a1.astype(float), axis=0, with_mean=True, with_std=True, copy=True )
        relu  = ReLU()
        z1 = relu.forward(a1)

        lt2 = LinearTransform(self.W2, self.B2)
        a2 = lt2.forward(z1)

        sce = SigmoidCrossEntropy()
        yhat = sce.forward(a2)

        # loss = (-1 / m) * (np.multiply(y, np.log(yhat)) + np.multiply((1 - y), np.log(1 - yhat)))
        pred = np.zeros((yhat.shape))
        pred[yhat >= 0.5] = 1
        pred[yhat < 0.5] = 0
        print(pred.shape)

        pos_log = np.log(yhat + 1e-15)
        neg_log = np.log(1 - yhat + 1e-15)
        loss =  (np.multiply(y, pos_log) + np.multiply((1 - y), neg_log)) 
        loss = -1*np.mean(loss, axis = 0) 

        return pred, loss

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):
	# INSERT CODE for training the network
        lt = LinearTransform(self.W1, self.B1)
        # print("x : ", x_batch)
        a1 = lt.forward(x_batch)
        # print("a1 : ", a1)
        # print(np.amax(a1), " : ", np.amin(a1))
        a1 = scale( a1.astype(float), axis=0, with_mean=True, with_std=True, copy=True )

        relu  = ReLU()
        z1 = relu.forward(a1)
        # z1 = normalize( z1)
        # print("z1 : ", z1)

        
        lt2 = LinearTransform(self.W2, self.B2)
        a2 = lt2.forward(z1)
        # return
        # print("a2 : ", a2)
        # relu1  = ReLU()
        # z2 = relu1.forward(a2)
        # print("z2 : ", z2)
        sce = SigmoidCrossEntropy()
        yhat = sce.forward(a2)
        # print("yhat : ", yhat)
        # print("y_batch : ", y_batch)
        finalLoss = self.calculate_loss(yhat, y_batch, l2_penalty)
        # finalLoss = (-1 / m) * (np.multiply(y_batch, np.log(yhat)) + np.multiply((1 - y_batch), np.log(1 - yhat)))
        # print("Final Loss : ", finalLoss)

        delta2 = sce.backward(y_batch)
        # print("delta : ", delta2.shape)
        # relu1_back = relu1.backward(delta)
        # print("relu1_back : ", relu1_back.shape)
        dW2, dB2 = lt2.backward(delta2)
  
        # print("W2 : ", self.W2.shape)
        # print("B2 : ", self.B2.shape)
        
        # print("dw2 : ", dW2.shape)
        # print("db2 : ", dB2.shape)
     


        self.mdW2 = momentum * self.mdW2 - learning_rate * (dW2  + l2_penalty * self.W2)
        self.W2 = self.W2 + self.mdW2

        self.mdB2 = momentum * self.mdB2 - learning_rate * np.sum(dB2)
        self.B2 = self.B2 + self.mdB2
        # print("W2 : ", self.W2.shape)
        # print("B2 : ", self.B2.shape)
        
        # return

        dz = delta2 * self.W2
        # print("dz : ", dz.shape)
        dr1 = relu.backward(dz)
        # print("dr1 : ", dr1.shape)
        dW1, dB1 = lt.backward(dr1)
        # print("dw1 : ", dW1.shape)
        # print("db1 : ", dB1.shape)
        # print("dw1 : ", np.sum(dW1).shape)
        # print("db1 : ", np.sum(dB1, axis = 0).shape )

        # print("W1 : ", self.W1.shape)
        # print("B1 : ", self.B1.shape)

        self.mdW1 = momentum * self.mdW1 - learning_rate * (dW1 + l2_penalty * self.W1)
        self.W1 = self.W1 + self.mdW1

        self.mdB1 = momentum * self.mdB1 - learning_rate * np.sum(dB1, axis = 0).reshape(self.B1.shape[0], 1)
        self.B1 = self.B1 + self.mdB1
        # print("W1 : ", self.W1.shape)
        # print("B1 : ", self.B1.shape)

        pred = (yhat >= 0.5).astype(int)
        return finalLoss, pred
        
    def evaluate(self, x, y):
        yhat, loss = self.predict(x, y)
        # print(yhat)
        prediction = np.zeros((yhat.shape))
        prediction[yhat == y] = 1
        return np.sum(prediction), loss

def get_random_array(min, max, size, is_with_replacement):
    # print(min, " : ", max, " : ", size)
    return np.random.choice(range(min, max), size, replace=is_with_replacement)

def group_list(l, group_size):
    batch_list = []

    for i in range(0, len(l), group_size):
        batch_list.append(l[i:i+group_size])
    
    return batch_list

def getBatches(x, y, batch_size):
    return group_list(x, batch_size), group_list(y, batch_size)




if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data[b'train_data']
    train_y = data[b'train_labels']
    test_x = data[b'test_data']
    test_y = data[b'test_labels']
    train_x = scale( train_x.astype(float), axis=0, with_mean=True, with_std=True, copy=True ).astype(float)

    test_x = scale( test_x.astype(float), axis=0, with_mean=True, with_std=True, copy=True ).astype(float)

    num_examples, input_dims = train_x.shape
	# INSERT YOUR CODE HERE-
    
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 100
    num_batches = 1000
    hidden_units = 128
    batch_size = 64
    m = len(train_x)
    train_acc = []
    test_acc = []
    batch_sizes = [64, 128, 256]
    learning_rate_list = [0.1, 0.01, 0.001]
    hidden_unit_list = [64, 128, 256]
    momentum_list = [0.1, 0.5, 0.8]
    l2_penalty_list = [0.1, 0.001, 0.0001]

    for i in range(1):
        tr_acc = []
        te_acc = []
        mlp = MLP(input_dims, 128)

        for epoch in range(num_epochs):

	    # INSERT YOUR CODE FOR EACH EPOCH HERE
            randArray = get_random_array(0, train_x.shape[0], train_x.shape[0], False)

            t_x_list, t_y_list = getBatches(train_x[randArray, :], train_y[randArray, :], 64)
            train_accuracy = 0
            total_loss = 0.0

            for b in range(len(t_x_list)):
	    		# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
	    		# MAKE SURE TO UPDATE total_loss
                loss, yhat = mlp.train(t_x_list[b], t_y_list[b], 0.001, 0.1, 0.1)
                # break

                result = yhat == t_y_list[b]
                # print(result)
                res = np.sum(result.astype(int))
                train_accuracy += res
                # print("train_accuracy : ", res / len(t_y_list[b]))
                # print("loss : ", loss)
                total_loss += loss

                # break

            print('\r[Epoch {}, mb {}]    Avg.Loss = {}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss / len(t_x_list),
                ),
                end='',
            )
            sys.stdout.flush()
            # break
                # break
	    	# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
	    	# MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
            train_accuracy, train_loss = mlp.evaluate(train_x, train_y)
            # tr_acc.append(train_accuracy)

            test_accuracy, test_loss = mlp.evaluate(test_x, test_y)
            print(test_accuracy)
            te_acc.append((test_accuracy / len(test_y)) * 100)
            # print()
            print("Train Loss: ", train_loss, "  Train Acc.: ", ((train_accuracy / m) * 100.0))
            print("Test Loss: ", test_loss, "  Test Acc.: ", ((test_accuracy / len(test_x)) * 100.0))
        test_acc.append(te_acc)
        print("Max Acc : ", max(te_acc))
        # train_acc.append(tr_acc)

    

    plt.figure()
    # plt.xticks(np.arange(0, len(train_accuracy), 5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    # print("L2 : ", len(hidden_unit_list))
    for j in range(len(hidden_unit_list)):
        # l = "l2 : " + str(hidden_unit_list[j])
        plt.plot(test_acc[j], label = str(learning_rate_list[j]))
    plt.legend()
    plt.savefig('AvsE_final.jpg')
