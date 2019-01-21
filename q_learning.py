#maxwell Sanders
#this is the implementation of the qlearning algorithm

import sys
import numpy as np

def q_learning_update (q,ntable,rows,row,col,nrow,ncol,g,N,nr, r, a):
	#check if it is terminal
	if(rows[nrow][ncol] != "."):
		q[nrow][ncol]["me"] = nr
	if row is not None and col is not None:
		if a in ntable[row][col]:
			ntable[row][col][a] += 1
		else: 
			ntable[row][col][a] = 1
		c = 1/ntable[row][col][a]
		if a not in q[row][col]:
			q[row][col][a] = 0
		q[row][col][a] = (1-c)*q[row][col][a]
		q[row][col][a] += c*(r + g*findmax(q,nrow,ncol))
	return q,ntable			

def findmaxnone(q,nrow,ncol):
	#get all the q directions
	u = None
	if "up" in q[nrow][ncol]:
		u = q[nrow][ncol]["up"]
	if "down" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["down"]
		if(u is None):
			u = tempu
		elif(tempu > u):
			u = tempu
	if "left" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["left"]
		if(u is None):
			u = tempu
		elif(tempu > u):
			u = tempu
	if "right" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["right"]
		if(u is None):
			u = tempu
		elif(tempu > u):
			u = tempu
	if "me" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["me"]
		if(u is None):
			u = tempu
		elif(tempu > u):
			u = tempu
	if(u is None):
		u = nont	
	if(rows[nrow][ncol] == "X"):
		u = 0
	return u



def findmax(q,nrow,ncol):
	#get all the q directions
	if "up" in q[nrow][ncol]:
		u = q[nrow][ncol]["up"]
	else:
		u = 0
	if "down" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["down"]
	else:
		tempu = 0
	if(tempu > u):
		u = tempu
	if "left" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["left"]
	else:
		tempu = 0
	if(tempu > u):
		u = tempu
	if "right" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["right"]
	else:
		tempu = 0
	if(tempu > u):
		u = tempu
	if "me" in q[nrow][ncol]:
		tempu = q[nrow][ncol]["me"]
	else:
		tempu = 0
	if(tempu > u):
		u = tempu
	return u

def getbestaction(q, ntable, N, nrow, ncol):
	#get all the q directions
	if "up" in q[nrow][ncol]:
		upq = q[nrow][ncol]["up"]
	else:
		upq = 0
	if "down" in q[nrow][ncol]:
		downq = q[nrow][ncol]["down"]
	else:
		downq = 0
	if "left" in q[nrow][ncol]:
		leftq = q[nrow][ncol]["left"]
	else:
		leftq = 0
	if "right" in q[nrow][ncol]:
		rightq = q[nrow][ncol]["right"]
	else:
		rightq = 0

	#get all the n directions
	if "up" in ntable[nrow][ncol]:
		upn = ntable[nrow][ncol]["up"]
	else:
		upn = 0
	if "down" in ntable[nrow][ncol]:
		downn = ntable[nrow][ncol]["down"]
	else:
		downn = 0
	if "left" in ntable[nrow][ncol]:
		leftn = ntable[nrow][ncol]["left"]
	else:
		leftn = 0
	if "right" in ntable[nrow][ncol]:
		rightn = ntable[nrow][ncol]["right"]
	else:
		rightn = 0

	#get the maximum utility action
	if(upn < N):
		u = 1
	else:
		u = upq
	a = "up"

	if(downn < N):
		tempu = 1
	else:
		tempu = downq
	
	if(tempu > u):
		a = "down"
		u = tempu
		
	if(leftn < N):
		tempu = 1
	else:
		tempu = leftq
	
	if(tempu > u):
		a = "left"
		u = tempu
		
	if(rightn < N):
		tempu = 1
	else:
		tempu = rightq
	
	if(tempu > u):
		a = "right"
		u = tempu
		
	return a


def executeaction(a, rows, row, col):
	#assign the winning direction index
	d = np.random.rand(1)
	if(d < .1):
		d = 2
	elif(d < .2):
		d = 1
	else:
		d = 0

	#assign directions
	if(a == "up"):
		temp = ["up","right","left"]	
	elif(a == "down"):
		temp = ["down","right","left"]	
	elif(a == "left"):
		temp = ["left", "up", "down"]	
	elif(a == "right"):
		temp = ["right","up","down"]	

	#try the action
	a = temp[d]

	if(a == "down"):
		if(row+1 == len(rows)):
			nrow = row
			ncol = col
		elif(rows[row+1][col]  == "X"):
			nrow = row
			ncol = col
		else:
			nrow = row + 1
			ncol = col
	if(a == "up"):
		if(row == 0):
			nrow = row 
			ncol = col
		elif(rows[row-1][col]  == "X"):
			nrow = row 
			ncol = col
		else:
			nrow = row - 1
			ncol = col
	if(a == "right"):
		if(col+1 == len(rows[0])):
			nrow = row
			ncol = col
		elif(rows[row][col+1]  == "X"):
			nrow = row
			ncol = col
		else:
			nrow = row 
			ncol = col + 1
	if(a == "left"):
		if(col == 0):
			nrow = row
			ncol = col
		elif(rows[row][col-1]  == "X"):
			nrow = row
			ncol = col
		else:
			nrow = row 
			ncol = col - 1

	return nrow, ncol

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

#get the variables from the command line
g = float(sys.argv[3])
nont = float(sys.argv[2])
N = int(sys.argv[5])

#instantiate the q-table and the Ntable
q = []
ntable = []
for i in range(height):
	ntable.append([])
	q.append([])
	for j in range(width):
		ntable[i].append({})
		q[i].append({})

#go through the updates based on the final argument
count = 0
moves = int(sys.argv[4])
while(True):
	row = None
	col = None
	r = None
	a = None
	
	nrow = int(np.random.rand(1)*height)
	ncol = int(np.random.rand(1)*width)
	#find a starting state
	while(rows[nrow][ncol] != "."):
		nrow = int(np.random.rand(1)*height)
		ncol = int(np.random.rand(1)*width)	
	#print('start = (',nrow,',',ncol,')')
	#go for a mission
	while(count < moves):
		count += 1
		
		#sense reward
		if(rows[nrow][ncol] == "."):
			nr = nont
		else:
			nr = float(rows[nrow][ncol])

		#q learning update
		q,ntable = q_learning_update(q,ntable,rows,row,col,nrow,ncol,g,N,nr, r,a)

		#check if nrow, ncol is terminal
		if(rows[nrow][ncol] != "."):
			break
		
		#determine the best action
		a = getbestaction(q,ntable,N,nrow,ncol)

		#execute the action
		row = nrow
		col = ncol
		r = nr
		nrow, ncol = executeaction(a,rows,nrow,ncol)
		#print('moving ',a,' to state (',nrow,',',ncol,')')	
	if(count >= moves):
		break

#print results
for i in range(height):
        #formatted string
        fstr = ""
        for j in range(width):
                fstr += "%6.3f," % findmaxnone(q,i,j)
        print(fstr[:-1])	
