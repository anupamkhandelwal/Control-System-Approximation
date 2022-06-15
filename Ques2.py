
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


# In[18]:

u = np.random.randn(1000,2)
x = np.zeros((1000,2))


# In[19]:

for i in range(1,u.shape[0]):
    x[i,0] = (x[i-1,0]/(1+x[i-1,1]*x[i-1,1])) + u[i-1,0]
    x[i,1] = ((x[i-1,1]*x[i-1,0])/(1+x[i-1,1]*x[i-1,1])) + u[i-1,1]


# In[4]:

def initialize_parameters(n_x, n_h1, n_h2, n_y):
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros(shape=(n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros(shape=(n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros(shape=(n_y, 1))
    
    parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2,
                    "W3": W3,
                 "b3":b3}
    
    return parameters


# In[5]:

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    return Z


# In[6]:

def linear_activation_forward(A_prev, W, b):
    Z = linear_forward(A_prev, W, b)
    A = 1/(1+np.exp(-Z))
    
    cache = ((A_prev, W, b), Z)
    
    return A, cache


# In[7]:

def compute_cost(AL, Y):
    
    m = Y.shape[0]

    cost = (1/m)*np.sum((AL-Y)**2)
    
    cost = np.squeeze(cost)      
    
    return cost


# In[8]:

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims=True)/ m
    dA_prev = np.dot(cache[1].T, dZ)
    
    return dA_prev, dW, db


# In[9]:

def linear_activation_backward(dA, cache):
    
    linear_cache, activation_cache = cache
    Z = activation_cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    # Shorten the code
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[10]:

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters


# In[11]:

def two_layer_model(X, Y, layers_dims, learning_rate=0.003, num_iterations=3000, print_cost=False):
    
    grads = {}
    costs = [] 
    m = X.shape[1] 
    (n_x, n_h1, n_h2, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    for i in range(0, num_iterations):

        A1, cache1 = linear_activation_forward(X, W1, b1)
        A2, cache2 = linear_activation_forward(A1, W2, b2)
        A3, cache3 = linear_activation_forward(A2, W3, b3)
        
        cost = compute_cost(A3, Y)
        
        dA3 = - (np.divide(Y, A3) - np.divide(1 - Y, 1 - A3))
        
        dA2, dW3, db3 = linear_activation_backward(dA3, cache3)
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2)
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1)
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW3'] = dW3
        grads['db3'] = db3
        
        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        W3 = parameters["W3"]
        b3 = parameters["b3"]
        
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



# In[12]:

params1 = two_layer_model(x[:999,:].T,x[1:,0].T,layers_dims=(2,10,10,1),learning_rate=0.003,num_iterations=10000,print_cost=True)


# In[13]:

params2 = two_layer_model(x[:999,:].T,x[1:,1].T,layers_dims=(2,10,10,1),learning_rate=0.003,num_iterations=10000,print_cost=True)


# In[14]:

def forward_pass(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    A1, cache1 = linear_activation_forward(X, W1, b1)
    A2, cache2 = linear_activation_forward(A1, W2, b2)
    A3, cache3 = linear_activation_forward(A2, W3, b3)
    
    return A3


# In[20]:

y = forward_pass(x[:999,:].T,parameters=params1)


# In[21]:

plt.rcParams["figure.figsize"] = (14,8)
fig, ax = plt.subplots()
plt.ylabel('Predicted output')
plt.xlabel('Time')
line1, = ax.plot(range(999), np.transpose(x[1:,0].T),color='red')
#line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(range(999), np.transpose((y+u[:999,0])),color = 'green')

plt.show()


# In[ ]:

plt.rcParams["figure.figsize"] = (20,10)

plt.scatter(x[1:,0],(y+u[:999,0]).T)


# In[22]:

y = forward_pass(x[:999,:].T,parameters=params2)
plt.rcParams["figure.figsize"] = (14,8)
fig, ax = plt.subplots()
plt.ylabel('Predicted output')
plt.xlabel('Time')
line1, = ax.plot(range(999), np.transpose(x[1:,1].T),color='red')
#line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax.plot(range(999), np.transpose((y+u[:999,1])),color = 'green')

plt.show()

