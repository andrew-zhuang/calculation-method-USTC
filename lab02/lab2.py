import numpy as np


def function(x):
    return 1.0 / (x ** 2 + 1)


def function_2(x, num):
    return -5 + 10 * x / num


def function_3(x, num):
    return -5 * np.cos(np.pi * (2 * x + 1) / (2 * num + 2))


def lagrange(x_train, y_train, number):
    x_test = np.arange(-5, 5, step=0.02)
    y_test = function(x_test)
    loss = np.empty(x_test.shape, dtype=float)
    for k in range(x_test.shape[0]):
        ans = 0.0
        for i in range(number):
            temp = y_train[i]
            for j in range(number):
                if i != j:
                    temp *= (x_test[k] - x_train[j]) / (x_train[i] - x_train[j])
            ans += temp
        loss[k] = abs(ans - y_test[k])
    return np.max(loss)


n = 41  # break point
initial_x = np.arange(0, n, dtype=float)
input_x1 = function_2(initial_x, n - 1)
input_y1 = function(input_x1)
input_x2 = function_3(initial_x, n - 1)
input_y2 = function(input_x2)
print("list1:")
print(5, lagrange(input_x1, input_y1, 5))
print(10, lagrange(input_x1, input_y1, 10))
print(20, lagrange(input_x1, input_y1, 20))
print(40, lagrange(input_x1, input_y1, 40))
print("list2:")
print(5, lagrange(input_x2, input_y2, 5))
print(10, lagrange(input_x2, input_y2, 10))
print(20, lagrange(input_x2, input_y2, 20))
print(40, lagrange(input_x2, input_y2, 40))
