import numpy as np
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return np.sqrt(np.mean(((np.array(y_pred) - np.array(y_true)) ** 2)))


def fit(X, Y):
    cov = 0
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    for i in range(len(X)):
        cov += (X[i] - x_mean) * (Y[i] - y_mean)
    b1 = cov / sum([(x - x_mean) ** 2 for x in X])
    b0 = y_mean - b1*x_mean
    return b0, b1


def predict(X, b0, b1):
    return [b0 + b1*x for x in X]


def create_dataset(b0, b1, size):
    X = range(size)
    Y = [b0 + b1 * x + np.random.randint(-size // 10, size // 10) for x in X]
    return X,  Y


if __name__ == "__main__":
    b0 = 10
    b1 = 0.4
    X, Y = create_dataset(b0, b1, 300)
    plt.scatter(X, Y)
    y_pred = predict(X, *fit(X, Y))
    y_true = [b0 + b1*x for x in X]
    print(rmse(y_true, y_pred))
    plt.plot(X, y_pred, c="r", linewidth=2)
    plt.plot(X, y_true, c="g")
    plt.show()
