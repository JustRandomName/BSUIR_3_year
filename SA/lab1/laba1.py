import numpy as np
import matplotlib.pyplot as plt

POST_POINT = 6

data = np.loadtxt('lab1-file', delimiter='		')
first_value = np.array([y for y, z in data])
second_value = np.array([z for y, z in data])

print(f' мат ожидание х = {round(np.mean(first_value), POST_POINT)}')
print(f' мат ожидание y = {round(np.mean(second_value), POST_POINT)}')

print(f' дисперсия х = {round(np.var(second_value), POST_POINT)}')
print(f' дисперсия y = {round(np.var(second_value), POST_POINT)}')

print(f' СКО х = {round(np.std(second_value), POST_POINT)}')
print(f' СКО у = {round(np.std(second_value), POST_POINT)}')

print(
    f'коэф корреляции К = {round(np.corrcoef(first_value, second_value)[0, 1], POST_POINT)}')

sorted_X = sorted(first_value)
sorted_Y = sorted(second_value)

plt.figure(figsize=(38.4, 21.6))

for i in range(len(sorted_X)):
    plt.scatter(first_value[i], second_value[i])

A = np.vstack([first_value, np.ones(len(first_value))]).T
m, c = np.linalg.lstsq(A, second_value, rcond=None)[0]

resulr_sorted_Y = [m * x + c for x in first_value]

plt.plot(first_value, resulr_sorted_Y)

X = [[1, x, x * x, x ** 3, x ** 4] for x in first_value]
X = np.array(X)
Y = np.array(second_value)

theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
X = [[1, x, x * x, x ** 3, x ** 4] for x in sorted_X]
X = np.array(X)
plt.plot(sorted_X, np.dot(X, theta_best))

print(theta_best)
plt.show()

print(first_value.size)
print(second_value.size)
