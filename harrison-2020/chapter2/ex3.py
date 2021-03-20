import numpy as np

a = [1, 2, 3]
b = [2, 3, 4] 

c = np.array([b]).T
print(c)

output = np.dot(a, b)
print(output)

output = np.dot(a, c)
print(output)
