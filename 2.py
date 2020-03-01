import numpy as np

# a = '0612.txt'
# print(int(a.split('.')[0]) + 513)
# b = int(a.split('.')[0]) + 513
# c = str(b).zfill(4)
# print(type(c), c)

a = np.random.rand(2, 3)
print(a)
for i, j, k in a:
    print(i, j, k)
