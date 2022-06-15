
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
ypredict=np.zeros([1,1000])
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


def setParameters(X, Y, hidden_size=10):
    
    input_size = X.shape[0] # number of neurons in input layer
    output_size = Y.shape[0] # number of neurons in output layer.
    W1 = np.random.randn(hidden_size, input_size)*np.sqrt(2/input_size)
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size)*np.sqrt(2/hidden_size)
    b2 = np.zeros((output_size, 1))
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def forwardPropagation(X, params):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(params['W2'], A1)+params['b2']
    y = Z2  
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def cost(predict, actual):
    m = actual.shape[1]
    cost__ = (1/1000)*np.sum((predict-actual)**2)
    for index in range(1000):
        ypredict[0][index] = predict[0][index]
    return np.squeeze(cost__)

def backPropagation(X, Y, params, cache):
    m = X.shape[1]
    dy = cache['y'] - Y
    dW2 = (1 / m) * np.dot(dy, np.transpose(cache['A1']))
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    dZ1 = np.dot(np.transpose(params['W2']), dy) * (1-np.power(cache['A1'], 2))
    dW1 = (1 / m) * np.dot(dZ1, np.transpose(X))
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def updateParameters(gradients, params, learning_rate = 1.2):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
 
def fit(X, Y, learning_rate, hidden_size=10, number_of_iterations = 5000):
    params = setParameters(X, Y, hidden_size)
    cost_ = []
    for j in range(number_of_iterations):
        y, cache = forwardPropagation(X, params)
        costit = cost(y, Y)
        gradients = backPropagation(X, Y, params, cache)
        params = updateParameters(gradients, params, learning_rate)
        cost_.append(costit)
        print(costit)
    return params, cost_

# Testing the code
u = np.random.randn(1000,1)
y_prev = 0
y = []
for i in u:
    y_prev = y_prev/(1+y_prev**2) + i**3
    y.append(y_prev)
    
train_x = np.zeros([2,1000])

for p in range(1000):
    train_x[0][p] = u[p]
    if p-1<0:
        
        train_x[1][p] = 0
        
    else:
        
        train_x[1][p] = y[p-1] 

        
train_y = np.transpose(np.array(y))



params, cost_ = fit(train_x, train_y, 0.03, 100, 10000)
import matplotlib.pyplot as plt
plt.plot(cost_)


# In[6]:

plt.ylabel('Mean square cost')
plt.xlabel('No. of iterations')
plt.plot(cost_[1000:])
plt.savefig('costp1nn.png')


# In[ ]:

ypredict.shape


# In[18]:

plt.rcParams["figure.figsize"] = (14,8)

fig, ax = plt.subplots()
plt.ylabel('Predicted output')
plt.xlabel('Time')
line1, = ax.plot(range(1000), np.transpose(train_y),color='red')
line2, = ax.plot(range(1000), np.transpose(ypredict),color='green'  )

plt.show()
plt.savefig('trainp1nn.png')


# In[19]:

u1 = np.random.randn(1000,1)
y_prev = 0
ytest = []
for i in u1:
    y_prev = y_prev/(1+y_prev**2) + i**3
    ytest.append(y_prev)
    
test_x = np.zeros([2,1000])

for p in range(1000):
    test_x[0][p] = u1[p]
    if p-1<0:
        
        test_x[1][p] = 0
        
    else:
        
        test_x[1][p] = ytest[p-1] 

        
test_y = np.transpose(np.array(ytest))


# In[21]:

y_pred, cache_pred = forwardPropagation(test_x, params)
plt.rcParams["figure.figsize"] = (14,8)
fig, ax = plt.subplots()
plt.ylabel('Predicted output')
plt.xlabel('Time')
line1, = ax.plot(range(1000), np.transpose(test_y),color='red')
line2, = ax.plot(range(1000), np.transpose(y_pred),color = 'green')

plt.show()


# In[22]:

cost_test = cost(y_pred, test_y)


# In[23]:

print(cost_test)


# In[ ]:



