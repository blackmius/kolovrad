import numpy as np
from kolovrad import Kolovrad

k = Kolovrad([2, 2, 1])

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

k.fit(X, Y)

print('Xor example', k)
print('''
Recall the truth table
X0 X1 Y
0  0  0
0  1  1
1  0  1
1  1  0\n''')

while True:
    x0, x1 = map(float, input('x0 x1: ').split())
    X = np.array([[x0, x1]])
    print(k.predict(X)[0, 0])
