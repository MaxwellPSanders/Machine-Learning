#Maxwell Sanders
#neural networks

import numpy as np
import sys

#the sigmoid function
def sigmoid(n):
	return 1/(1+np.exp(-n))

#np.random.seed(1)

#open up the training file
traindata = np.loadtxt(sys.argv[1])
testdata = np.loadtxt(sys.argv[2])

#get all of the classes
classes = np.unique(traindata[:,-1]).tolist()

#print(classes)

#get all of the system inputs
numlayers = int(sys.argv[3])
perlayer = int(sys.argv[4])
rounds = int(sys.argv[5])

#print(perlayer)

#normalize the data
#print(traindata)
traindata = traindata[:,:-1]
traindata /= np.max(traindata)
testdata = testdata[:,:-1]
testdata /= np.max(testdata)
#print(traindata)
testclass = np.loadtxt(sys.argv[2])[:,-1]

#since we chopped off the classes off of training, let's get the expected back
trainexpect = np.loadtxt(sys.argv[1])[:,-1]

#print(trainexpect)

#initialize the weights
z = []
w = []
d = []

lowbound = -0.5
highbound = 0.5
#append the first row
if(numlayers < 2):
	print("This is not possible, exiting")
	exit()
if(numlayers == 2):
	for i in range(traindata.shape[1] + 1):
		w.append([np.random.uniform(low=lowbound,high=highbound,size=len(classes))])
if(numlayers > 2):
	#for the first row
	for i in range(traindata.shape[1] + 1):
		w.append([np.random.uniform(low=lowbound,high=highbound,size=perlayer)])
	#for all of the inner rows
	for i in range(numlayers - 3):
		for j in range(perlayer + 1):
			w.append([np.random.uniform(low=lowbound,high=highbound,size=perlayer)])
	#for the final output
	for i in range(perlayer + 1):
		w.append([np.random.uniform(low=lowbound,high=highbound,size=len(classes))])

#create the layers matrix
layers = []
index = 0
#for the first layer
layers.append([])
for i in range(traindata.shape[1] + 1):
	layers[0].append(i)
	z.append(1)
	d.append(0)
#for the middle layers
for i in range(numlayers - 2):
	layers.append([])
	for j in range(perlayer + 1):
		index = traindata.shape[1] + 1 + j + i*(perlayer+1)
		layers[i+1].append(index)	
		z.append(1)
		d.append(0)
#for the final layer
layers.append([])
for i in range(len(classes)):
	layers[-1].append(index+i+1)
	z.append(1)
	d.append(0)	

#print(classes)
#print(layers)

#for i in range(len(w)):
#	print(str(i) + " " + str(w[i])) 

#learning rate
learn = 1

#go through every round
for run in range(rounds):
	#go through every data point
	for x in range(traindata.shape[0]):
		#calculate the output of each layer
		for i in range(len(layers)):
			#if it is the first layer, the output is just the input
			if( i == 0):
				for j in range(traindata.shape[1]):





					#this line will need to be changed
					z[j+1] = traindata[x][j]
					





			#check if it is the last layer
			elif( i == len(layers) - 1):
				#go through every element in the layer except bias
				for j in layers[i]:
					z[j] = 0
					#go through every element of the previous layer
					for k in layers[i-1]:
						z[j] += z[k]*w[k][0][layers[i].index(j)]
						#print(j,k,w[k],w[k][0][0])
					z[j] = sigmoid(z[j])
			#if it is every other layer
			else:
				#go through every element in the layer except bias
				for j in layers[i][1:]:
					z[j] = 0
					#go through every element of the previous layer
					for k in layers[i-1]:
						z[j] += z[k]*w[k][0][layers[i].index(j)-1]
						#print(j,k,w[k],w[k][0][0])
					z[j] = sigmoid(z[j])
			
			#print(z)

		#start from the final layer
		for i in range(len(layers) - 1,-1,-1):
			#print(i)
			#for the output layer
			if( i == len(layers) - 1):
				for j in layers[i]:



					#this line will need to be changed
					if(classes.index(trainexpect[x]) == layers[i].index(j)):




						expected = 1
					else:
						expected = 0
					d[j] = (z[j] - expected)*z[j]*(1-z[j])
					#print(j,d[j], expected)

			#for every other layer
			else:
				#for every node in that layer
				for j in layers[i]:
					#print(j, w[j][0])
					totaldiff = 0
					#through every weight in that layer
					if( i == len(layers) - 2):
						for k in range(len(layers[i+1])):
							totaldiff += d[layers[i+1][k]]*w[j][0][k]
					else:
						for k in range(len(layers[i+1]) - 1):
							totaldiff += d[layers[i+1][k]]*w[j][0][k]
					#print(j,d[j])
					d[j] = totaldiff*z[j]*(1-z[j])
					#print(j,d[j])


		#go through all the weights and update them
		#for every layer except the first
		#print(layers)
		for l in range(1, len(layers), 1):
			#for every perceptron
			if(l == len(layers) - 1):
				for j in layers[l]:
					#for every perceptron in the previous layer
					for i in layers[l-1]:
						w[i][0][layers[l].index(j)] = w[i][0][layers[l].index(j)] - learn*d[j]*z[i]
					#	print(j, i, w[i][0])
			else:
				for j in layers[l][:-1]:
					#print(j)
					#for every perceptron in the previous layer
					for i in layers[l-1]:
						w[i][0][layers[l].index(j)] = w[i][0][layers[l].index(j)] - learn*d[j]*z[i]
	

	#update the learning factor
	learn *= .98

	#print(d)

total_accuracy = 0
#start running through the test cases
#go through every data point
for x in range(testdata.shape[0]):
	#calculate the output of each layer
	for i in range(len(layers)):
		#if it is the first layer, the output is just the input
		if( i == 0):
			for j in range(traindata.shape[1]):





				#this line will need to be changed
				z[j+1] = testdata[x][j]






		#check if it is the last layer
		elif( i == len(layers) - 1):
			#go through every element in the layer except bias
			for j in layers[i]:
				z[j] = 0
				#go through every element of the previous layer
				for k in layers[i-1]:
					z[j] += z[k]*w[k][0][layers[i].index(j)]
					#print(j,k,w[k],w[k][0][0])
				z[j] = sigmoid(z[j])
		#if it is every other layer
		else:
			#go through every element in the layer except bias
			for j in layers[i][1:]:
				z[j] = 0
				#go through every element of the previous layer
				for k in layers[i-1]:
					z[j] += z[k]*w[k][0][layers[i].index(j-1)]
					#print(j,k,w[k],w[k][0][0])
				z[j] = sigmoid(z[j])

	#largest output
	largep = layers[-1][0]
	#go through the output perceptrons
	for i in layers[-1]:
		#print(x,i,z[i],z[largep])
		if(z[largep] < z[i]):
			#print(x,i,z[i],z[largep])
			largep = i
	#print(x,largep)
	accuracy = 0	
	if(classes[layers[-1].index(largep)] == testclass[x]):
		total_accuracy += 1
		accuracy = 1
	print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f"% (x + 1, classes[layers[-1].index(largep)], testclass[x], accuracy))
print("classification accuracy=%6.4f" % (total_accuracy/testdata.shape[0]))
