#maxwell Sanders
#KNN Classify

import numpy as np
import sys
from collections import Counter
import random

def main():
	#get in the number of clusters
	k = int(sys.argv[3])

	#read in the training dataset
	train = np.loadtxt(sys.argv[1])
	trainclass = train[:,-1]
	train = train[:,:-1]
	#read in the testing dataset
	test = np.loadtxt(sys.argv[2])
	testclass = test[:,-1]
	test = test[:,:-1]
	
	#number of classes
	numclass = np.unique(trainclass).shape[0]

	#total accuracy
	total_accuracy = 0

	#get the mean and stddev of training dataset
	m = np.mean(train, axis=0)
	s = np.std(train, axis=0)
	#normalize the training dataset
	train -= m
	train /= s
	#normalize the testing dataset
	test -= m
	test /= s
	#go through the testing data point by point
	for i in range(test.shape[0]):
		#create what will hold the cluster data
		nodes = [None]
		distances = [None]
		for x in range(k-1):
			nodes.append(None)
			distances.append(None)
		#go through the training data point by point
		for j in range(train.shape[0]):
			distance = 0
			#go through every column and get the distance
			for x in range(train.shape[1]):
				distance += (train[j][x] - test[i][x])**2
			distance = np.sqrt(distance)

			#check to see if this is the lowest one yet
			for x in range(k):
				#if nothing is there add it
				if(distances[x] is None):
					distances[x] = distance
					nodes[x] = trainclass[j]
					break
				#if the thing there is worse then insert before it and
				#pop off the last element of the list
				if(distances[x] > distance):
					distances.insert(x,distance)
					distances = distances[:-1]
					nodes.insert(x,trainclass[j])
					nodes = nodes[:-1]
					break
		#print(distances, nodes)		
		#look for ties and then then determine the winner
		counts = Counter(nodes).most_common(numclass)
		highest = counts[0][1]
		best = []
		ties = 0
		for x in counts:
			if x[1] == highest:
				ties += 1
				best.append(x[0])
			else:
				break

		#get the stats to print out
		predicted_class = random.choice(best)
		true_class = testclass[i]
		if(predicted_class == true_class):
			accuracy = 1/ties
			total_accuracy += accuracy
		else:	
			accuracy = 0
		print("ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f" % (i+1, predicted_class, true_class, accuracy))


	#print the final accuracy
	print("classification accuracy=%6.4f" % (total_accuracy/test.shape[0]))
	return

main()
