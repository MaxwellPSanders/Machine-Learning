#Maxwell Sanders
#1001069652
#kmeans clustering algorithm

import sys
import numpy as np

#calculate the error for a set
def error(data, center):
	total_error = 0
	#go through every data point in the set
	for x in range(len(data)):
		lerror = 0
		#go through every column
		for y in range(data[x].shape[0]):
			err = (data[x][y] - center[y])**2
			lerror += err
		total_error += np.sqrt(lerror)
	return total_error
		

#The clustering function, it will accept a dataset and the return the new centers
def clustering(data, centers):
	#create the new variables
	new_centers = []
	clusters = []
	#create a list for every cluster
	for x in range(len(centers)):
		clusters.append([])	

	#go through each center data point
	for x in range(data.shape[0]):	
		cluster = 0	
		#go through every center and decide which center it should belong to
		for y in range(len(centers)):
			#current center error
			cdist = 0
			#go through every column
			for z in range(data.shape[1]):
				cdist += (data[x][z] - centers[y][z])**2
			#and then finally calculate the distance
			cdist = np.sqrt(cdist)
			
			#if it is the first center then give it the smallest error and index
			if(y == 0):
				dist = cdist
			elif(cdist < dist):
				dist = cdist
				cluster = y
		#add to the total error and at to the cluster
		clusters[cluster].append(data[x])

	#go through the clusters to create the new centers
	for x in range(len(clusters)):
		new_centers.append(np.mean(clusters[x],axis=0))

	return new_centers, clusters

#This is the main function
def main():
	#get the data and set the variables
	data = np.loadtxt(sys.argv[1])[:,:-1]
	clusters = int(sys.argv[2])

	#create the centers by randomly assigning variables and getting the means
	#shuffle the dataset
	np.random.shuffle(data)	
	centers = []

	terror = 0
	#go for each centers
	for x in range(clusters):
		#append a center created by averaging a partition from the data
		begin = int(0+x*(data.shape[0]/clusters))
		end = int((x+1)*(data.shape[0]/clusters))
		centers.append(np.mean(data[begin:end],axis=0))
		terror += error(list(data[begin:end]),centers[x])
		#print("Center: ",x+1,centers[x]) 		
	print("After initialization: error = %.4f" % terror)	

	#go through each iteration	
	for x in range(int(sys.argv[3])):
		#call the clustering algorithm
		centers, clusters = clustering(data, centers)	

		terror = 0
		for y in range(len(clusters)):
			err = error(clusters[y],centers[y])
			terror += error(clusters[y],centers[y])
			#print("Center: ",y+1,centers[y]) 		
		print("After iteration %d: error = %.4f" % (x+1,terror))

main()
