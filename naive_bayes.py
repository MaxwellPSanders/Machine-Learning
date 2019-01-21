#Maxwell Sanders
#Naive Bayes classifier

import sys
import numpy
from decimal import *
import math

#create a function that calculates P(x|C)
def formula(x, mean, std):
	exponent = Decimal(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
	exponent = exponent.exp()
	return Decimal(1 / (math.sqrt(2*math.pi) * std)) * exponent

#open up the training file
training = open(sys.argv[1], "r")
test = open(sys.argv[2], "r")
t_lines = training.readlines()

#check to see how it is laid out
elements = len(t_lines[0].split())

#create an array for the classes
classes = []	
counts = []
p_class = []

#go through each line to get the correct amount of classes
for line in t_lines:
	#break it into elements
	tokens = line.split()

	#check if the class is already in the array
	if(int(tokens[-1]) not in classes):
		classes.append(int(tokens[-1]))
		counts.append(1)
	else:
		counts[classes.index(int(tokens[-1]))] = counts[classes.index(int(tokens[-1]))] + 1
		

classes, counts = (list(t) for t in zip(*sorted(zip(classes, counts))))
#print(classes)

means = []
stds = []
#go back through the whole thing for each class
for x in range(len(classes)):
	means.append([])
	stds.append([])
	#get the counts and probabilities
	p_class.append(counts[x]/numpy.sum(counts))
	#print(p_class[x])
	
	attributes = []
	#create attributes
	for y in range(elements - 1):
		attributes.append([]) 

	for line in t_lines:
		#break the line into tokens
		tokens = line.split()
		
		#check to see if that line is the class we are looking for
		if(int(tokens[-1]) ==  classes[x]):
			for z in range(elements - 1):
				attributes[z].append(float(tokens[z]))
	#print(attributes)				
	
	#go through the attributes
	for y in range(len(attributes)):
		classs = int(classes[x])
		means[x].append(numpy.mean(attributes[y]))
		stds[x].append(numpy.std(attributes[y]))
		if(stds[x][y] < .01):
			stds[x][y] = .01
		print("Class %d, attribute %d, mean=%.2f, std=%.2f" % (classs, y+1, means[x][y], stds[x][y]))

#go ahead and open up
tests = test.readlines()

accuracies = []	
#go through all the test cases
for x in range(len(tests)):
	tokens = tests[x].split()	

	results = []
	#go through all of the classes
	for y in range(len(classes)):
		p_x_c = 1
		for z in range(elements - 1):
			#add on the formula
			p_x_c *= Decimal(formula(float(tokens[z]),means[y][z],stds[y][z]))
			#print(float(tokens[z]), means[y][z],stds[y][z],p_x_c)
		#print(p_x_c)
		#append another slot onto the results
		results.append(p_x_c * Decimal(p_class[y]))
	#sum all of the results together to get p_x
	p_x = numpy.sum(results)
	
	#go through all of the results and then divide p_x
	for y in range(len(results)):
		results[y] /= p_x
		#print(results[y])

	ties = []
	largest = -1
	index = 0
	#go back through the results and figure out which one is the largest
	for y in range(len(results)):
		#determine the largest number
		if(largest < results[y]):
			largest = results[y]
			ties = [classes[y]]
			index = y
		elif(largest == results[y]):
			ties.append(classes[y])
	
	accuracy = 0
	#check if it is true
	if(int(tokens[-1]) in ties):
		true = True
		accuracy = 1/len(ties)
	accuracies.append(accuracy)	 

	#after going through the array and picking the largest one, print out the result
	print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (x + 1, classes[index], results[index], int(tokens[-1]), accuracy))
	
print("classification accuracy=%6.4f" % numpy.mean(accuracies))	
