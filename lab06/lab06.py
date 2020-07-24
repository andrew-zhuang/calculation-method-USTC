import numpy as np
import matplotlib.pyplot as plt


class GaussSeidel:
    def __init__(self, A, b, x_value, eps=1e-9, iterations=400):
        self.A = np.array(A)
        self.b = b
        self.eps = eps
        self.iterations = iterations
        self.results = x_value
        self.G = None
        self.g = None
        self.D = None
        self.L = None
        self.U = None

    def warming(self):
        D = np.zeros((self.A.shape[0], self.A.shape[0]))
        L = np.zeros((self.A.shape[0], self.A.shape[0]))  # lower
        U = np.zeros((self.A.shape[0], self.A.shape[0]))  # upper
        for i in range(D.shape[0]):
            D[i][i] = self.A[i][i]
            for j in range(i):
                L[i][j] = -self.A[i][j]
                U[self.A.shape[0] - 1 - i][self.A.shape[0] - 1 - j] = \
                    -self.A[self.A.shape[0] - 1 - i][self.A.shape[0] - 1 - j]
        self.G = np.dot(np.linalg.inv(D - L), U)
        self.g = np.dot(np.linalg.inv(D - L), self.b)
        self.D = D
        self.L = L
        self.U = U

    def calculate(self):
        self.warming()
        x_old = self.results
        for i in range(self.iterations):
            x_new = np.dot(self.G, x_old) + self.g
            if np.linalg.norm(x_new - x_old) < self.eps:
                self.results = x_new
                print("iteration: ", i)
                print("x: ", x_new)
                return
            else:
                x_old = x_new
        print("Having reach the max iterations, the solution is not a converges one! ")


class SOR:
    def __init__(self, A, b, x_value, omega, eps=1e-6, iterations=20000):
        self.A = A
        self.b = b
        self.result = x_value
        self.omega = omega
        self.eps = eps
        self.iterations = iterations
        self.iteration = 0
        self.D = None
        self.L = None
        self.U = None
        self.B = None
        self.f = None

    def warming(self):
        D = np.zeros((self.A.shape[0], self.A.shape[0]))
        L = np.zeros((self.A.shape[0], self.A.shape[0]))  # lower
        U = np.zeros((self.A.shape[0], self.A.shape[0]))  # upper
        for i in range(D.shape[0]):
            D[i][i] = self.A[i][i]
            for j in range(i):
                L[i][j] = -self.A[i][j]
                U[self.A.shape[0] - 1 - i][self.A.shape[0] - 1 - j] = \
                    -self.A[self.A.shape[0] - 1 - i][self.A.shape[0] - 1 - j]
        self.D = D
        self.L = L
        self.U = U
        self.B = np.dot(np.linalg.inv(D - self.omega * L), (1 - self.omega) * D + omega * U)
        self.f = self.omega * np.dot(np.linalg.inv(D - self.omega * L), self.b)

    def calculation(self):
        self.warming()
        x_old = self.result
        for i in range(self.iterations):
            x_new = np.dot(self.B, x_old) + self.f
            if np.linalg.norm(x_new - x_old) < self.eps:
                self.result = x_new
                self.iteration = i
                return
            else:
                x_old = x_new
        print("Having reach the max iterations, the solution is not a converges one! ")


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    initial = np.array([[31, -13, 0, 0, 0, -10, 0, 0, 0],
                        [-13, 35, -9, 0, -11, 0, 0, 0, 0],
                        [0, -9, 31, -10, 0, 0, 0, 0, 0],
                        [0, 0, -10, 79, -30, 0, 0, 0, -9],
                        [0, 0, 0, -30, 57, -7, 0, -5, 0],
                        [0, 0, 0, 0, -7, 47, -30, 0, 0],
                        [0, 0, 0, 0, 0, -30, 41, 0, 0],
                        [0, 0, 0, 0, -5, 0, 0, 27, -2],
                        [0, 0, 0, -9, 0, 0, 0, -2, 29]])
    b = np.array([-15, 27, -23, 0, -20, 12, -7, 7, 10])
    initial_x = np.ones(initial.shape[0])
    GS = GaussSeidel(initial, b, initial_x)
    GS.calculate()
    omega_list = np.linspace(0.02, 1.98, 99)
    result = []
    for omega in omega_list:
        SOR_ = SOR(initial, b, initial_x, omega)
        SOR_.calculation()
        print("relaxing factor: ", omega, "\titeration: ", SOR_.iteration)
        result.append(SOR_.iteration)
        del SOR_

    result = np.array(result)
    plt.title("Relation between relaxing factor and iterations")
    plt.plot(omega_list, result)
    plt.savefig(r"./result.png")
    plt.show()
