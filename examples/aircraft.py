import numpy as np
from kolovrad import Kolovrad

k = Kolovrad([2, 3, 2])

# speed, weight

X = np.array([
    [2414, 8573],
    [2414, 19700],
    [2656 , 12701],

    [2460, 6800 ],
    [2000, 9850 ],
    [2500, 16380],
    [2400 , 11000],

    [2485, 19838],
    [2237, 8725],

    [1900, 110000],
    [1100, 13381],
    [1010, 71700],
    [1335, 87090],
    [1654, 22300],
    [1510, 85000],
    [1047, 83250],
    [830, 90000],
    [558, 36850],
    [197, 2000],

    [785, 10000]
])

# normalize
X = X / np.linalg.norm(X, np.inf)

print(X)

# fighter, bomber

Y = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],
    [1, 0],

    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],

    [0, 1], 

    [1, 1]
])

k.fit(X, Y)

print('Aircraft example', k)
print('For given speed and weight trying to guess if it is fighter nor bomber\n')

while True:
    x0, x1 = map(float, input('speed weight: ').split())
    X = np.array([[x0, x1]])
    print(np.round(k.predict(X), decimals=3))
