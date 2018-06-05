import tensorflow as tf
import numpy as np
import pandas as pd

#size of each batch
batchsize=100
#number of output classes
n_class = 10
#number of iterations of training
epochs = 45

#loading training data
dataset = pd.read_csv('train.csv',sep=',')
X = dataset.iloc[:,1:].values
#normalizing the input features
X=X/255
#output vector
y_ini = dataset.iloc[:,0].values
#converting the output vector into one hot matrix
Y = np.zeros((y_ini.size,n_class))
for i,j in enumerate(y_ini):
	Y[i,j]=1

#placeholders for input and output data
x = tf.placeholder('float')
y = tf.placeholder('float')

#declaring weights and biases
weights = {'cl1':tf.Variable(tf.random_normal([5,5,1,6])),
		   'cl2':tf.Variable(tf.random_normal([5,5,6,16])),
		   'fl1':tf.Variable(tf.random_normal([7*7*16,120])),
		   'fl2':tf.Variable(tf.random_normal([120,84])),
		   'outl':tf.Variable(tf.random_normal([84,n_class]))}
biases = {'cl1':tf.Variable(tf.random_normal([6])),
		  'cl2':tf.Variable(tf.random_normal([16])),
		  'fl1':tf.Variable(tf.random_normal([120])),
		  'fl2':tf.Variable(tf.random_normal([84])),
		  'outl':tf.Variable(tf.random_normal([n_class]))}

#feedforward of the network
def convnet(x):
	#reshaping the input into 4D matrix where each example is a 28x28x1 3D matrix
	x = tf.reshape(x,[-1,28,28,1])
	#first convolutional layer with 6 5x5x1 filters and stride=1 
	conv1 = tf.nn.leaky_relu(tf.nn.conv2d(x,weights['cl1'],strides=[1,1,1,1],padding="SAME",use_cudnn_on_gpu=False)+biases['cl1'])
	#average pooling with window size 2x2 and stride=2
	conv1 = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#second convolutional layer with 16 5x5x6 filters and stride=1
	conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1,weights['cl2'],strides=[1,1,1,1],padding="SAME",use_cudnn_on_gpu=False)+biases['cl2'])
	#average pooling with window size 2x2 and stride=2
	conv2 = tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#ravelling the 4D volumke into 2D matrix where each row is the activations of respective example
	conv2 = tf.reshape(conv2,[-1,7*7*16])
	#fully connected layer with relu activation
	fcl1 = tf.nn.leaky_relu(tf.matmul(conv2,weights['fl1'])+biases['fl1'])
	#fully connected layer with relu activation
	fcl2 = tf.nn.leaky_relu(tf.matmul(fcl1,weights['fl2'])+biases['fl2'])
	#output layer
	output = tf.matmul(fcl2,weights['outl'])+biases['outl']
	return output

def train():
	#predicting the output using feedforward
	prediction = convnet(x)
	#cost function
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,logits=prediction))
	#Optimization function. here Adam is used
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	
	#starting a Tensorflow session
	with tf.Session() as sess:
		#initializing the declared weights and biases
		sess.run(tf.global_variables_initializer())
		
		#training the network 
		for epoch in range(epochs):
			#cost for each epoch
			epochcost=0.0
			
			#dividing the large training set into batches
			for k in  range(0,int(X.shape[0]/batchsize)):
				batchx,batchy=X[k*batchsize:k*batchsize+batchsize],Y[k*batchsize:k*batchsize+batchsize]
				#running the optimizer and cost function
				_,c=sess.run([optimizer,cost],feed_dict={x:batchx,y:batchy})
				#adding the cost of each batch to the epoch cost
				epochcost+=c
			
			#printing the number of epochs completed and it's respective cost
			print('Epoch',epoch+1,'of',epochs,'epochs. Loss is ',epochcost)
		
		#saving weights for later use if needed
		np.save('conv1w.npy',weights['cl1'].eval())
		np.save('conv2w.npy',weights['cl2'].eval())
		np.save('full1w.npy',weights['fl1'].eval())
		np.save('full2w.npy',weights['fl2'].eval())
		np.save('outlw.npy',weights['outl'].eval())
		#saving biases
		np.save('conv1b.npy',biases['cl1'].eval())
		np.save('conv2b.npy',biases['cl2'].eval())
		np.save('full1b.npy',biases['fl1'].eval())
		np.save('full2b.npy',biases['fl2'].eval())
		np.save('outlb.npy',biases['outl'].eval())
		
		#Loading the test set
		dataset_test = pd.read_csv('mnist_test.csv',sep=',')
		#input features
		X_test = dataset_test.iloc[:,1:].values
		#normalizing input features
		X_test=X_test/255
		#output vector
		y_test_ini = dataset_test.iloc[:,0].values
		#converting to one hot matrix
		Y_test = np.zeros((y_test_ini.size,n_class))
		for i,j in enumerate(y_test_ini):
			Y_test[i,j]=1
		#calculating accuracy
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:X_test, y:Y_test}))


train()
