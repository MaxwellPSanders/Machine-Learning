#Maxwell Sanders
#Kmeans clustering

import numpy as np
import sys

#np.random.seed(1)

class Node:
	def __init__(self,nid=None,feature=-1,threshold=-1,gain=0, distribution=None):
		self.nid = nid
		self.feature = feature
		self.threshold = threshold
		self.gain = gain
		self.distribution = distribution
		self.left = None
		self.right = None

def print_tree(root, tid):
	if root is None: return
	queue = []
	queue.append(root)

	while(len(queue))> 0:
		print('tree=%2d, node=%3d, feature=%2d, thr=%6.2f, gain=%f' % (tid, queue[0].nid, queue[0].feature, queue[0].threshold, queue[0].gain))
		#print(queue[0].distribution)
		node = queue.pop(0)
		if node.left is not None: queue.append(node.left)
		if node.right is not None: queue.append(node.right)

#infogain
def infogain(examples, avalues, threshold):
	classes = np.unique(examples[:,-1])
	classes = np.append(classes, classes[-1] + 1)
	#print(classes)
	#get the distribution of everything
	dist = (np.histogram(examples[:,-1], classes)[0])/examples.shape[0]
	#print(dist)
	#print(dist)	
	#calculate the entropy of the starting node
	ent = 0
	for i in range(dist.shape[0]):
		ent += -(dist[i])*np.log2(dist[i])

	#print(dist)		
	#print()
	
	left = []
	right = []
	#determine how many go left and right
	for i in range(examples.shape[0]):
		if avalues[i] < threshold:
			left.append(examples[i,-1])
		else:
			right.append(examples[i,-1])

	#print(left)
	#print(right)

	#get the distribution of these distributions and their entropies
	dleft = (np.histogram(left, classes)[0])/len(left)	
	#print(dleft)
	info = ent
	for i in range(dleft.shape[0]):
		if(dleft[i] != 0):
			info -= (len(left)/examples.shape[0])*(-(dleft[i])*np.log2(dleft[i]))

	dright = (np.histogram(right, classes)[0])/len(right)
	for i in range(dright.shape[0]):
		if(dright[i] != 0):
			info -= (len(right)/examples.shape[0])*(-(dright[i])*np.log2(dright[i]))

#	print(len(left)+len(right),examples.shape[1])
#	print(info)
#	print()

	return info 

#this is the optimized cha cha which you do da
def optimized(examples,attributes):
	max_gain = -1
	best_attribute = -1
	best_threshold = -1
	#go through each attribute
	for i in attributes:
		#print(i)
		a_values = examples[:,i]
		L = np.min(a_values)
		M = np.max(a_values)
		#go through the thresholds
		for j in range(51)[1:]:
			threshold = L + j*(M-L)/51
			gain = infogain(examples,a_values,threshold)
			if gain > max_gain:
				max_gain = gain
				best_attribute = i
				best_threshold = threshold
	
	#return everything
	#print(best_attribute, best_threshold)
	return best_attribute,best_threshold, max_gain

def randomized(examples,attributes):
	max_gain = -1
	best_attribute = -1
	best_threshold = -1
	a = np.random.choice(attributes)
	#print(i)
	a_values = examples[:,a]
	L = np.min(a_values)
	M = np.max(a_values)
	#go through the thresholds
	for j in range(51)[1:]:
		threshold = L + j*(M-L)/51
		gain = infogain(examples,a_values,threshold)
		if gain > max_gain:
			max_gain = gain
			best_attribute = a
			best_threshold = threshold
	
	#return everything
	#print(best_attribute, best_threshold)
	return best_attribute,best_threshold, max_gain

#top level function that will call the recursive function to build the tree
def DTL_toplevel(examples, pruning, random):
	#get the classes
	classes = np.unique(examples[:,-1])
	classes = np.append(classes, classes[-1] + 1)
	#get the attributes 
	attributes = list(range(examples.shape[1] - 1))
	#print(attributes) 
	#print(random)
	#get the default class
	default = (np.histogram(examples[:,-1], classes)[0])/examples.shape[0]
	#print(default)
	#print(extra)
	return DTL(examples,attributes,default,pruning,classes,random,1)

def DTL(examples, attributes, default, pruning,classes,random,nid):
	#if there are no examples
	if examples.shape[0] < pruning:
		return Node(nid=nid,distribution=default)
	#if there are examples but they are all the same class
	elif (np.unique(examples[:,-1]).shape[0] == 1):
		dist = []
		for i in classes[:-1]:
			if(i == examples[0,-1]):
				dist.append(1)
			else:
				dist.append(0)
		return Node(nid=nid, distribution=dist)
	#create a split
	else:
		distro = (np.histogram(examples[:,-1],classes)[0])/examples.shape[0]
		#choose an attribute based off of whether we want this to be random or not
		if random:
			best_attribute, best_threshold, gain = randomized(examples,attributes)
		else:
			best_attribute, best_threshold, gain = optimized(examples,attributes)
		#create the node
		tree = Node(nid=nid, distribution=default, threshold=best_threshold, gain=gain, feature=best_attribute)
		lexamples = examples[examples[:,best_attribute]<best_threshold]
		rexamples = examples[examples[:,best_attribute]>=best_threshold]
		#print(nid, best_attribute, best_threshold)
		tree.left = DTL(lexamples, attributes, distro, pruning, classes, random, nid*2)
		tree.right = DTL(rexamples, attributes, distro, pruning, classes, random, nid*2 + 1)
		#split the elements
		return tree		

#test the trees return the index of the classes
def test_trees(trees, row):
	adistribution = 0
	#go through each of the trees
	for i in range(len(trees)):
		node = trees[i]
		#for the first tree set the adistribution
		if(i == 0):
			#keep going until you reach the root
			while(True):
				if(node.feature == -1):
					adistribution = node.distribution
					break
				elif(row[node.feature] > node.threshold):
					node = node.right
				else:
					node = node.left	
		else:
			#keep going until you reach the root
			while(True):
				if(node.feature == -1):
					adistribution = np.add(adistribution,node.distribution)
					break
				elif(row[node.feature] > node.threshold):
					node = node.right
				else:
					node = node.left	
			
	for i in range(len(adistribution)):
		adistribution[i] /= i+1	

	#find the highest and randomize ties
	high = max(adistribution)
	indices = []
	for i in range(len(adistribution)):
		if(adistribution[i] == high):
			indices.append(i)
	
	return indices

#open up the training file
traindata = np.loadtxt(sys.argv[1])
testdata = np.loadtxt(sys.argv[2])

#store the other flags
option = sys.argv[3]
pruning = float(sys.argv[4])

if option == "optimized":
	#call DTL toplevel
	trees = [DTL_toplevel(traindata,pruning,0)]
	print_tree(trees[0],1)
elif option == "randomized":
	trees = [DTL_toplevel(traindata,pruning,1)]
	print_tree(trees[0],1)
elif option == "forest3":
	trees = []
	for i in range(3):
		trees.append(DTL_toplevel(traindata,pruning,1))
		print_tree(trees[i],i+1)
elif option == "forest15":
	trees = []
	for i in range(15):
		trees.append(DTL_toplevel(traindata,pruning,1))
		print_tree(trees[i], i+1)
#test the tree
accuracy = 0
classes = np.unique(traindata[:,-1])
#for every row
for i in range(testdata.shape[0]):
	#get the index of the winning class
	indices = test_trees(trees, testdata[i,:])			
	#resolve ties
	index = np.random.choice(indices)
	#deterimine if it was accurate 
	if(classes[index] == testdata[i,-1]):
		accuracy += 1
		print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f'% (i+1, classes[index], testdata[i,-1],1/len(indices)))
	else:
		print('ID=%5d, predicted=%3d, true=%3d, accuracy=%4.2f'% (i+1, classes[index], testdata[i,-1],0))
	

#get the total accuracy
accuracy /= testdata.shape[0]

#print out the total accuracy
print('classification accuracy=%6.4f'% accuracy)
