import numpy as np
import scipy as sp
from scipy import integrate


class VariationalCalculus:
    def __init__(self, points_list, left, right, function):
        """
        :param function: sin
        :param points_list: [0, 1, 2, /cdots, 12)]
        :param left: 1
        :param right: 5
        """
        self.left = left
        self.right = right
        self.__point_list = points_list
        self.function = function
        self.__points_number = None
        self.points = None
        self.values = None
        self.Trapezoid = None
        self.Simpson = None
        self.eps_Trapezoid = None
        self.eps_Simpson = None
        self.eps_index_Trapezoid = None
        self.eps_index_Simpson = None

    def __init_points(self):
        self.__points_number = [pow(2, i) for i in self.__point_list]
        for i in range(1, len(self.__point_list)+1):
            self.__points_number.append(i * self.__points_number[i - 1])

    def __convert_points(self):
        def calculate(number):
            # print(number)
            return list(np.linspace(self.left, self.right, int(number+1)))
        self.points = list(map(calculate, iter(self.__points_number)))

    # the function for integral
    def __function(self):
        self.values = [self.function(list_points) for list_points in self.points]

    def __Trapezoid_method(self):
        def calculate(value_list):
            length = len(value_list) - 1
            h = (self.right - self.left) / float(length)
            return h / 2 * (2 * sum(value_list) - value_list[0] - value_list[length])

        self.Trapezoid = [calculate(value_list) for value_list in self.values]

    def __Simpson_method(self):
        def calculate(value_list):
            length = len(value_list) - 1
            h = (self.right - self.left) / float(length)
            odds_list = value_list[1::2]
            even_list = value_list[::2]
            return h / 3 * (2 * sum(even_list) + 4 * sum(odds_list) - value_list[0] - value_list[length])
        self.Simpson = [calculate(value_list) for value_list in self.values]

    def __eps_Trapezoid(self):
        integral_result = integrate.quad(self.function, self.left, self.right)
        # print(integral_result)
        self.eps_Trapezoid = list(map(lambda s: integral_result[0] - s, iter(self.Trapezoid)))

    def __eps_Simpson(self):
        integral_result = integrate.quad(self.function, self.left, self.right)
        self.eps_Simpson = list(map(lambda s: integral_result[0] - s, iter(self.Simpson)))

    def __eps_Trapezoid_index(self):
        length = len(self.eps_Trapezoid)
        real_length = int(length / 2)
        self.eps_index_Trapezoid = []
        for i in range(real_length):
            res = np.log(np.abs(self.eps_Trapezoid[i] / self.eps_Trapezoid[i + real_length])) / (np.log(i + 1))
            self.eps_index_Trapezoid.append(res)

    def __eps_Simpson_index(self):
        length = len(self.eps_Simpson)
        real_length = int(length / 2)
        self.eps_index_Simpson = []
        for i in range(real_length):
            res = np.log(np.abs(self.eps_Simpson[i] / self.eps_Simpson[i + real_length])) / (np.log(i + 1))
            self.eps_index_Simpson.append(res)

    def solve(self):
        self.__init_points()
        self.__convert_points()
        self.__function()
        self.__Trapezoid_method()
        self.__eps_Trapezoid()
        self.__eps_Trapezoid_index()
        self.__Simpson_method()
        self.__eps_Simpson()
        self.__eps_Simpson_index()


if __name__ == '__main__':
    number_list = list(np.linspace(0, 12, 13))
    variational = VariationalCalculus(number_list, 1, 5, np.sin)
    variational.solve()
    print("复化梯形积分公式的误差和误差阶为：")
    for i in range(len(number_list)):
        print(i, variational.eps_Trapezoid[i], variational.eps_index_Trapezoid[i])
    print("复化Simpson积分公式的误差和误差阶为：")
    for i in range(len(number_list)):
        print(i, variational.eps_Simpson[i], variational.eps_index_Simpson[i])
    print("done\n")
