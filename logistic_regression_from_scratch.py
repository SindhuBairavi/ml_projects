import pandas as pd
import numpy as np

Y = np.sort(np.random.randint(low=0, high=2, size=1000))
X1_0 = np.random.normal(loc=0, scale=1, size=500)
X1_1 = np.random.normal(loc=5, scale=1, size=500)
X1 = np.hstack((X1_0, X1_1))
X2_0 = np.random.normal(loc=0, scale=1, size=500)
X2_1 = np.random.normal(loc=2, scale=1, size=500)
X2 = np.hstack((X2_0, X2_1))
# X = np.vstack((X1, X2))

df = pd.DataFrame([Y, X1, X2]).T
df.columns = ["labels", "X1", "X2"]
df["labels"] = df.labels.astype('int')
df.head(3)

X = df[["X1", "X2"]].values
y = df.labels.values


# Sample of 1000 entries in the form (Label, X1, X2)
#   -----------------------------------------------
#   |         |  Labels | X1          | X2        |
#   ----------|---------|-------------|------------
#   | 0       |  0.     |  0.295856   |  0.695861 |
#   | 1       |  0.     |  -2.805441  |  0.606429 |
#   | 2       |  1.     |  0.867072   |  -0.946759|
#   -----------------------------------------------
#

def prob(X, theta):
    return 1 / (1 + np.exp(-np.dot(X, theta)))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def fit(X, y):
    theta = np.ones(X.shape[1])
    learning_rate = 0.1
    epoch = 25
    for i in range(0, epoch):
        y_pred = prob(X, theta)
        grad = np.dot(X.T, (y_pred - y)) / y.size
        theta = theta - (grad * learning_rate)
        loss_value = loss(y_pred, y)
        print("Epoch #", i, ":", loss_value)


fit(X, y)
