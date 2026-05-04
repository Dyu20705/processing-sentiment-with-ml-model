import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, w, b):
    m = len(y)
    cost_sum = 0
    for i in range(m):
        z = np.dot(w, X[i]) + b
        g= sigmoid(z)
        cost_sum += -y[i] * np.log(g) - (1 - y[i]) * np.log(1 - g)

    return (1 / m) * cost_sum

def gradient_function(X, y, w, b):
    n = len(w)
    m = len(y)
    grad_w = np.zeros(n)
    grad_b = 0

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        grad_b += (g - y[i])
        for j in range(n):
            grad_w[j] += (g - y[i]) * X[i][j]

    grad_w /= m
    grad_b /= m

    return grad_w, grad_b

def gradient_descent(X, y, alpha, interations):
    w = np.zeros(n)
    b = 0

    for i in range(interations):
        grad_w, grad_b = gradient_function(X, y, w, b)
        w -= alpha * grad_w
        b -= alpha * grad_b

        if i % 100 == 0:
            cost = cost_function(X, y, w, b)
            print(f"Cost after iteration {i}: {cost}")

    return w, b

def predict(X, w, b):
    m = X.shape[0]
    pred = np.zeros(m)

    for i in range(m):
        z = np.dot(w, X[i]) + b
        g = sigmoid(z)
        pred[i] = 1 if g >= 0.5 else 0

    return pred
