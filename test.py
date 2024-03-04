import icon_benchmark
import numpy as np

x = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=float)

runtimes = icon_benchmark.nabla4_benchmark(x, 1, 1, 1, 1, 1)

print(runtimes)
