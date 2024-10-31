import numpy as np
import math

def H_theta(X, theta):
    h_theta = X @ theta 
    return h_theta

def J_theta(X, Y, theta):
    size = np.size(Y, 0)
    predicted = H_theta(X, theta)
    j_theta = (1/(2*size)) * np.transpose(predicted - Y) @ (predicted - Y)
    return j_theta

def gradient_descent(X, Y, iter=1000, alpha = 0.001):
    theta = np.zeros((1, np.size(X,1)))
    J_hist = np.zeros((iter, 2))
    m = np.size(Y)
    X_T = np.transpose(X) 
    J_hist[0,0] = 0
    J_hist[0,1] = np.round(J_theta(X, Y, theta[0]), 5)
    for i in range(1, iter):
        theta = theta - (alpha/(m))*(X_T @ (H_theta(X, theta[0]) - Y))
        compute_j_theta = np.round(J_theta(X, Y, theta[0]), 15)

        print('%r:%r'%(i,J_hist[i-1, 1]))
        if compute_j_theta - J_hist[i-1, 1] >= 0:
            return theta, J_hist
        else:
            J_hist[i,0] = i
            J_hist[i,1] = np.round(J_theta(X, Y, theta[0]), 15)
    return theta, J_hist

def normal_equalizion(X, Y):
    X_T = np.transpose(X)
    theta = (np.linalg.pinv(X_T@X))@(X_T@Y)
    return theta

def Normalize(X):
    X_temp = np.copy(X)
    X_temp[0,0] = 100
    _std = np.std(X, 0, dtype=np.float64)
    _mean = np.mean(X, 0)
    X_new = (X_temp-_mean)/_std
    X_new[:,0] = 1
    return X_new

raw = np.loadtxt('concac.txt', delimiter=';')

target = raw[:,6]
data = np.zeros((np.size(target), np.size(raw,1) ))

data[:,0] = 1
data[:,1:8] = raw[:,0:7]
data[:,8:] = raw[:,8:]

#data = Normalize(data)

# train 70%
Y_train = target[:int(np.size(target,0)*0.7)]
X_train = data[:int(np.size(target,0)*0.7),:]

# test 30%
Y_test = target[int(np.size(target,0)*0.7):np.size(target,0)]
X_test = data[int(np.size(target,0)*0.7):np.size(target,0),:]






