import numpy as np


class Gauss:
    def __init__(self, co_matrix):
        self.__matrix = co_matrix
        self.__matrix_after = None
        self.result = None

    def transfer(self):
        row = self.__matrix.shape[0]
        matrix = self.__matrix
        for i in range(row):
            # pre processing
            for k in range(i, row):
                for p in range(i, row):
                    if matrix[k][p] == 0:
                        continue
                    if matrix[k][p] < 0:
                        matrix[k] = matrix[k] * (-1)
                        break
                    else:
                        break
            # print(matrix)
            max_row = np.argmax(matrix[i:, i]) + i
            # print("max_row", max_row)
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            for j in range(row):
                if i == j:
                    continue
                matrix[j, :] = matrix[j, :] - matrix[i, :] * matrix[j, i] / matrix[i, i]
        self.__matrix_after = matrix

    def solve(self):
        print(self.__matrix_after)
        temp = self.__matrix_after
        last_column = temp.shape[1] - 1
        self.result = [temp[i][last_column] / temp[i][i] for i in range(temp.shape[0])]

    def call(self):
        self.transfer()
        self.solve()


if __name__ == "__main__":
    weight = np.array([[31, -13, 0, 0, 0, -10, 0, 0, 0],
                       [-13, 35, -9, 0, -11, 0, 0, 0, 0],
                       [0, -9, 31, -10, 0, 0, 0, 0, 0],
                       [0, 0, -10, 79, -30, 0, 0, 0, -9],
                       [0, 0, 0, -30, 57, -7, 0, -5, 0],
                       [0, 0, 0, 0, -7, 47, -30, 0, 0],
                       [0, 0, 0, 0, 0, -30, 41, 0, 0],
                       [0, 0, 0, 0, -5, 0, 0, 27, -2],
                       [0, 0, 0, -9, 0, 0, 0, -2, 29]])
    bias = np.array([[-15, 27, -23, 0, -20, 12, -7, 7, 10]])
    matrix = np.concatenate((weight, bias.T), axis=1)
    gauss = Gauss(matrix)
    gauss.call()
    for i in range(weight.shape[0]):
        print("%.6f" % (gauss.result[i]))
