#maxwell Sanders
#this is a logistic regressor

import numpy as np
import sys

np.set_printoptions(suppress=True)

def sigmoid(n):
	return 1/(1+np.exp(-n))

def predict(weightings, features):
	prediction = np.dot(features,weightings.T)
	prediction = sigmoid(prediction)
	return prediction

def creater(y):
	r = np.zeros(shape=(y.shape[0],y.shape[0]))
	for x in range(r.shape[0]):
		for z in range(r.shape[0]):
			if(x==z):
				r[x][z] = y[x][0]*( 1 - y[x][0])	
	return r

def getweights(weights, phi, t):	
	y = predict(weights.T, phi)
	r = creater(y)
#	print(weights.shape)
#	print(phi.shape)
#	print(t.shape)
#	print(y.shape)
#	print(r.shape)
	weights = weights - np.dot(np.dot(np.linalg.pinv(np.dot(np.dot(phi.T,r),phi)),phi.T),(y-t))
	return weights

def cecalc(weights,t,phi):
	p = predict(weights.T,phi)
	
	class0 = (1-t)*np.log(1-p)
	class1 = -t*np.log(p)

	ce = class0 - class1
	
	ce = ce.sum()/phi.shape[0]
	#print(ce)
	return ce

#open up the training file
degree = int(sys.argv[2])
traindata = np.loadtxt(sys.argv[1])
testdata = np.loadtxt(sys.argv[3])

#print(sigmoid(400),sigmoid(-400),sigmoid(0))
#print(traindata.shape)

#set the last column to the correct values
for x in range(traindata.shape[0]):
	if(traindata[x][traindata.shape[1] - 1] == 1):
		traindata[x][traindata.shape[1] - 1] = 1
	else:
		traindata[x][traindata.shape[1] - 1] = 0

#create the phi matrix
columns = traindata.shape[1] if degree == 1 else traindata.shape[1]*degree - 1
phi = np.ndarray(shape=(traindata.shape[0], columns), dtype=float)
#print(phi.shape)

weights = np.zeros(shape=(columns,1))

#get the t values
t = np.zeros(shape=(traindata.shape[0],1))
for x in range(traindata.shape[0]):
	t[x] = traindata[x][traindata.shape[1] - 1]

#go through each column and populate the data for the phi array
for x in range(traindata.shape[0]):
	phi[x][0] = 1	
	for i in range(traindata.shape[1] - 1):
		if(degree == 1):
			phi[x][i + 1] = traindata[x][i]
		else:
			phi[x][2*i + 1] = traindata[x][i]
			phi[x][2*i + 2] = traindata[x][i]**2

#go through until a break
i = 0
difference = 1
cedifference = 1
ce = cecalc(weights,t,phi)
while difference >= .001 and cedifference >= .001:
	i += 1
	newweights = getweights(weights,phi, t)
	difference = np.sum(np.absolute(newweights - weights))
	weights = newweights
	newce = cecalc(weights,t,phi)
	cedifference = newce - ce
	ce = newce
	#print (i, cedifference)

#print out the weights for the end of the test data
for x in range(weights.shape[0]):
	print("w%d=%.4f" % (x,weights[x][0]))

#go through all of the test lines and print the results
total_accuracy = 0
for x in range(testdata.shape[0]):
	object_id = x+1
	phi = np.zeros(shape=(1, columns))
	phi[0][0] = 1
	for i in range(testdata.shape[1] - 1):
		if(degree == 1):
			phi[0][i + 1] = testdata[x][i]
		else:
			phi[0][2*i + 1] = testdata[x][i]
			phi[0][2*i + 2] = testdata[x][i]**2
	predicted_class = 1 if predict(weights.T,phi) > .5 else 0
	if(predicted_class):
		probability = predict(weights.T,phi)
	else: 
		probability = 1 - predict(weights.T,phi)
	true_class = testdata[x][-1] if testdata[x][-1] == 1 else 0
	if(probability == .5):
		accuracy = .5
	elif(predicted_class == true_class):
		accuracy = 1
	else:
		accuracy = 0
	print('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f' % (object_id, predicted_class, probability, true_class, accuracy))
	total_accuracy += accuracy


total_accuracy /= testdata.shape[0]
print('classification accuracy=%6.4f' % (total_accuracy))
