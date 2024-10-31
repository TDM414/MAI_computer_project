import numpy as np


def H_theta(X, theta):
    h_theta = X @ theta 
    return h_theta

def J_theta(X, Y, theta):
    size = np.size(Y, 0)
    predicted = H_theta(X, theta)
    j_theta = (1/(2*size)) * np.transpose(predicted - Y) @ (predicted - Y)
    return j_theta

def gradient_descent(X, Y, theta, iner=1000, alpha = 0.001):
    theta = np.zeros((1, np.size(X,1)))
    J_hist = np.zeros((iner, 2))
    m = np.size(Y)
    X_T = np.transpose(X) 
    J_hist[0,0] = 0
    J_hist[0,1] = np.round(J_theta(X, Y, theta[0]), 5)
    for i in range(1, iner):
        theta = theta - (alpha/(1000*m))*(X_T @ (H_theta(X, theta[0]) - Y))
        cacul_j_theta = np.round(J_theta(X, Y, theta[0]), 15)
        if cacul_j_theta - J_hist[i-1, 1] == 0:
            print('a')
            return J_hist
        J_hist[i,0] = i
        J_hist[i,1] = np.round(J_theta(X, Y, theta[0]), 5)
        # J_hist[i,1] = J_theta(X, Y, theta[0])
    return J_hist

def normal_equalizion(X, Y):
    X_T = np.transpose(X)
    theta = (np.linalg.pinv(X_T@X))@(X_T@Y)
    return theta

raw = np.loadtxt('test.txt', delimiter=',')

Y = raw[:,2]
X = np.zeros((np.size(Y), np.size(raw,1)))
X[:,0] = 1
X[:,1:] = raw[:,:2]
theta = np.array([1,2,3])
# a = gradient_descent(X, Y, theta, iner=1000)
# np.savetxt('a.txt', a, delimiter=',')

theta = normal_equalizion(X, Y)
print(J_theta(X, Y, theta))
# print(J_theta(X, Y, theta=np.array([89597.909542,139.210674 ,-8738.019112])))