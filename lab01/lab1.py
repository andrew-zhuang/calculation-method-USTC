import numpy as np

input_x = np.array([0.0, 0.5, 1.0, np.sqrt(2), 10.0, 100.0, 300.0])
output_y = np.empty(input_x.shape, dtype=float)


for i in range(pow(10, 6)):
    output_y += 1.0 / ((i + 1 + input_x) * (i + 1))


for i in range(input_x.shape[0]):
    print(input_x[i], output_y[i])
