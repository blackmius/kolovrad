import numpy as np
from kolovrad import Kolovrad

k = Kolovrad([3, 3], learning_rate=0.01, activation='relu')

X = np.array([
    [1, 0, 0],
    [2, 25, 235],
    [3, 32, 62],
    [4, 1, 252],
    [5, 124, 53],
    [6, 124, 54],
    [7, 59, 90],
    [8, 49, 50],
    [255, 255, 255]
])
Y = np.array([
    [1, 0, 0],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
    [5, 0, 0],
    [6, 0, 0],
    [7, 0, 0],
    [8, 0, 0],
    [255, 0, 0]
])

k.fit(X, Y)

print('Red filter example', k)
print('Enter RGB value for Kolovrad and he will try reset green and blue values\n')

while True:
    x0, x1, x2 = map(float, input('r g b: ').split())
    X = np.array([[x0, x1, x2]])
    print(np.around(k.predict(X)))
    
