
# coding: utf-8

# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

inp_lay = 1000

inp = np.zeros((inp_lay,2))
u = np.random.randn(inp_lay,1)

for i in range(0,inp_lay):
	u[i] = math.sin(0.5*i) + math.cos(0.1*i)

inp[0][1] = u[0]

x1 = np.random.randn(inp_lay,1)
inp[0][0] = x1[0]

for i in range(1,inp_lay):
	x1[i] = x1[i-1]/(1+x1[i-1]*x1[i-1]) + u[i-1][0]
	inp[i] = x1[i]
	inp[i][1] = u[i]
	inp[i][0] = x1[i]
	

outp = np.zeros((inp_lay,1))

for i in range(0,inp_lay):
	outp[i] = x1[i-1]


plt.scatter(inp[:,0],inp[:,1])

k = 20
kmean = KMeans(n_clusters = k, max_iter = 3000)

kmean.fit(inp)


centr = kmean.cluster_centers_

print(centr)

plt.scatter(centr[:,0],centr[:,1],c='r')
plt.savefig('datarbftrain.png')
#plt.show()

def cost(y,yd):
	si = y.shape
	err = 0.0
	for i in range(0,inp_lay):
		err = err + (yd[i] - y[i])**2
		err = err/inp_lay
		return err


def dist(x,y):
	d = 0
	#si = x.shape
	#print(si.shape)
	for i in range(0,2):
		d = d + (x[i] - y[i])**2
	d = np.sqrt(d)
	return d

max_dist = 0
for i in range(0,k):
	for j in range(0,k):
		if(max_dist < dist(centr[i,:],centr[j,:])):
			max_dist = dist(centr[i,:],centr[j,:])

#print(max_dist)

sigma = max_dist/np.sqrt(2*k)

def rbf(x,y,sigma):
	d = dist(x,y)**2
	r = np.exp(-d/2*sigma**2)
	return r


alpha = 0.01


wi = np.random.randn(k,1)
#print(wi)

#err = np.zeros((15000))
error = np.zeros((15000))


for i in range(0,15000):
	y = 0
	for l in range(0,k):
		y = y + wi[l]*rbf(inp[i%inp_lay,:],centr[l,:],sigma)
	err = outp[i%inp_lay] - y
	for l in range(0,k):
		wi[l] = wi[l] + alpha*err*rbf(inp[i%inp_lay,:],centr[l,:],sigma)
	error[i] = err*err/2
	print(error[i]) 


#print(wi)
ypred = np.zeros((inp_lay))
for i in range(0,inp_lay):
	for l in range(0,k):
		ypred[i] = ypred[i] + wi[l]*rbf(inp[i,:],centr[l,:],sigma)

fig,ax = plt.subplots()


plt.plot(range(inp_lay),outp)
plt.plot(range(inp_lay),ypred)
plt.xlabel('Number of samples')
plt.ylabel('Actual Output/Predicted Output')
plt.title('Graph shows actual output and predicted values in Training')
plt.show()


plt.scatter(outp,ypred)
plt.xlabel('Predicted Output')
plt.ylabel('Actual Output')
plt.title('Graph shows actual output vs predicted values in Training')
plt.show()

#plt.show()

plt.plot(range(15000),error)
plt.xlabel('Number of Iterations')
plt.ylabel('Error')
plt.title('Cost')

plt.show()


#testing data


inp_lay = 100


inp = np.zeros((inp_lay,2))
u = np.random.randn(inp_lay,1)

for i in range(0,inp_lay):
	u[i] = math.sin(0.1*i)**2

inp[0][1] = u[0]

x1 = np.random.randn(inp_lay,1)
inp[0][0] = x1[0]

for i in range(1,inp_lay):
	x1[i] = x1[i-1]/(1+x1[i-1]*x1[i-1]) + u[i-1][0]
	inp[i] = x1[i]
	inp[i][1] = u[i]
	inp[i][0] = x1[i]
	

outp = np.zeros((inp_lay,1))

for i in range(0,inp_lay):
	outp[i] = x1[i-1]


ypred = np.zeros((inp_lay))
for i in range(0,inp_lay):
	for l in range(0,k):
		ypred[i] = ypred[i] + wi[l]*rbf(inp[i,:],centr[l,:],sigma)



plt.plot(range(inp_lay),outp)
plt.plot(range(inp_lay),ypred)
plt.xlabel('Number of samples')
plt.ylabel('Actual Output/Predicted Output')
plt.title('Graph shows actual output and predicted values in Testing')
plt.show()


plt.scatter(outp,ypred)
plt.xlabel('Predicted Output')
plt.ylabel('Actual Output')
plt.title('Graph shows actual output vs predicted values in Testing')
plt.show()



# In[ ]:



