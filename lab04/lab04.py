import numpy as np
from sympy import *


class NonlinearEquations:
    def __init__(self, initial, secant_init_1, secant_init_2, eps, function, function_derivative):
        self.initial = initial
        self.secant_init_1 = secant_init_1
        self.secant_init_2 = secant_init_2
        self.eps = eps
        self.function = function
        self.function_derivative = function_derivative
        self.Newton_solver = None
        self.Secant_solver = None

    def __Newton(self):
        def calculate(x_axis, level):
            x_axis = x_axis - self.function(x_axis) / self.function_derivative(x_axis)
            if self.function(x_axis) < self.eps:
                return x_axis, level
            return calculate(x_axis, level+1)
        self.Newton_solver = list(map(calculate, iter(self.initial), iter(np.ones(len(self.initial)))))

    def __Secant(self):
        def calculate(x_now, x_previous, level):
            x_after = x_now - self.function(x_now) / (self.function(x_now) - self.function(x_previous)) * (x_now - x_previous)
            if self.function(x_after) < self.eps:
                return x_after, level
            return calculate(x_after, x_now, level+1)
        self.Secant_solver = list(map(calculate, self.secant_init_1, self.secant_init_2, iter(np.ones(len(self.initial)))))

    def call(self):
        self.__Newton()
        self.__Secant()


if __name__ == "__main__":
    def f(x):
        return x ** 3 / 3 - x

    def f_derivative(x):
        return x ** 2 - 1

    Newton_init = [0.1, 0.2, 0.9, 9.]
    Secant_init = np.array([[0.0, 0.1], [0.1, 0.2], [0.8, 0.9], [8., 9.]])
    eps = 1e-5
    nonlinear_equations = NonlinearEquations(Newton_init, list(Secant_init[:, 0]),
                                             list(Secant_init[:, 1]), eps, f, f_derivative)
    nonlinear_equations.call()
    print("Newton迭代，初值、根和迭代步数：")
    for i in range(len(Newton_init)):
        print(Newton_init[i], nonlinear_equations.Newton_solver[i][0],
              nonlinear_equations.Newton_solver[i][1])
    print("弦截法，初值、根和迭代步数：")
    for i in range(len(Secant_init)):
        print(Secant_init[i][0], Secant_init[i][1],nonlinear_equations.Secant_solver[i][0],
              nonlinear_equations.Secant_solver[i][1])
    print("done")
