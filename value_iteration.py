#maxwell Sanders
#This is the value iteration algorithm

import sys

#this takes in the three values, checks to see if they are good and then sums them
def computeutil(one,two,three, me):
	util = 0
	if(one is not None):
		util += .8 * one
	else:
		util += .8 * me
	if(two is not None):
		util += .1 * two
	else:
		util += .1 * me
	if(three is not None):	
		util += .1 * three
	else:
		util += .1 * me
	return util

#take the original layout to find bounds, the u to find values, and row and column
def computemax(mat,u,r,c,g,nont):
	states = []
	north = None
	south = None
	east = None
	west = None
	#calculate what all the directions would take
	#north
	if(not (r+1 == len(mat) or mat[r+1][c] == "X")):
		north = u[r+1][c]

	#south
	if(not (r == 0 or mat[r-1][c] == "X")):
		south = u[r-1][c]
		
	#east 
	if(not (c+1 == len(mat[0]) or mat[r][c+1] == "X")):
		east = u[r][c+1]

	#west
	if(not (c == 0 or mat[r][c-1] == "X")):
		west = u[r][c-1]
	
	#me
	me = u[r][c]

	#then calculate what it would take to transition to those states
	#north
	states.append(computeutil(north,east,west,me))
	states.append(computeutil(south,east,west,me))
	states.append(computeutil(east,north,south,me))
	states.append(computeutil(west,north,south,me))


	return max(states)
		

#open the file and create the array
env = open(sys.argv[1], "r")

#split all of the rows
rows = env.readlines()

#split all of the columns
for i in range(len(rows)):
	rows[i] = rows[i].rstrip().split(",")

#get the heighth and width of the future utility array
height = len(rows)
width = len(rows[0])

#initialize everything
newU = []
for i in range(height):
	newU.append([])
	for j in range(width):
		newU[i].append(0)

oldU = newU

#get the variables from the command line
g = float(sys.argv[3])
nont = float(sys.argv[2])

#outer loop for number of iterations
for i in range(int(sys.argv[4])):
	newU = oldU
	
	#go through the array
	for j in range(height):
		for k in range(width):	
			#consult the map to see if we are on a blocked or terminal state
			if(rows[j][k] == "X"):
				continue
			elif(rows[j][k] != "."):
				newU[j][k] = float(rows[j][k])
			#if it isn't those then do the algorithm
			else:
				newU[j][k] = nont + g*computemax(rows,oldU,j,k,g,nont)
				
	oldU = newU

#print results
for i in range(height):
	#formatted string
	fstr = ""
	for j in range(width):
		fstr += "%6.3f," % oldU[i][j]
	print(fstr[:-1])
