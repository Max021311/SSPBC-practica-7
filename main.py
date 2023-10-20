import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./irisbin.csv', header=None)
X = np.asarray(df.loc[:, 0:5].values)
Y = np.asarray(df.loc[:, 6].values)
X.shape[0]

def validate(partitions: tuple[int, ...], X = np.array, Y = np.array):
  assert type(partitions) == tuple, 'Partitions must be a tuple of ints'
  assert sum(partitions) == 100, 'Sum of paritions must be 100'
  assert type(X) == np.ndarray, 'X must be a NDArray'
  assert type(Y) == np.ndarray, 'X must be a NDArray'
  assert X.shape[0] == Y.shape[0], f'X and Y must have the same dimension'

def split_sequencial (partitions: tuple[int, ...], X = np.array, Y = np.array):
  validate(partitions, X, Y)

  length = X.shape[0]
  start_index = 0
  for partition in partitions:
    step = round(partition / 100 * length)
    end_index = start_index + step
    yield X[start_index:end_index]
    yield Y[start_index:end_index]
    start_index = end_index
    start_index = end_index

accumulator = 0
subsets = split_sequencial((29, 40, 31), X, Y)
for index, partition in enumerate(subsets):
  accumulator += partition.shape[0]
  print(f'{"X" if index % 2 == 0 else "Y"}: {partition.shape}')
print(accumulator/2)

X_train, Y_train, X_test, Y_test = split_sequencial((51, 49), X, Y)

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# Graficar en cada subgráfico
axs[0, 0].plot(X_train.T[0], Y_train)
axs[0, 0].set_title('train X1 with Y')

axs[0, 1].plot(X_train.T[1], Y_train)
axs[0, 1].set_title('train X2 with Y')

axs[0, 2].plot(X_train.T[2], Y_train)
axs[0, 2].set_title('train X3 with Y')

axs[1, 0].plot(X_train.T[3], Y_train)
axs[1, 0].set_title('train X4 with Y')

axs[1, 1].plot(X_train.T[4], Y_train)
axs[1, 1].set_title('train X5 with Y')

axs[1, 2].plot(X_train.T[5], Y_train)
axs[1, 2].set_title('train X6 with Y')
# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# Graficar en cada subgráfico
axs[0, 0].plot(X_test.T[0], Y_test)
axs[0, 0].set_title('test X1 with Y')

axs[0, 1].plot(X_test.T[1], Y_test)
axs[0, 1].set_title('test X2 with Y')

axs[0, 2].plot(X_test.T[2], Y_test)
axs[0, 2].set_title('test X3 with Y')

axs[1, 0].plot(X_test.T[3], Y_test)
axs[1, 0].set_title('test X4 with Y')

axs[1, 1].plot(X_test.T[4], Y_test)
axs[1, 1].set_title('test X5 with Y')

axs[1, 2].plot(X_test.T[5], Y_test)
axs[1, 2].set_title('test X6 with Y')
# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()

def split_shuffle (partitions: tuple[int, ...], X = np.array, Y = np.array):
  X, Y = X.copy(), Y.copy()
  np.random.shuffle(X)
  np.random.shuffle(Y)
  return split_sequencial(partitions, X, Y)

accumulator = 0
subsets = split_shuffle((29, 40, 31), X, Y)
for index, partition in enumerate(subsets):
  accumulator += partition.shape[0]
  print(f'{"X" if index % 2 == 0 else "Y"}: {partition.shape}')

X_train, Y_train, X_test, Y_test = split_shuffle((31, 69), X, Y)

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# Graficar en cada subgráfico
axs[0, 0].plot(X_train.T[0], Y_train)
axs[0, 0].set_title('train X1 with Y')

axs[0, 1].plot(X_train.T[1], Y_train)
axs[0, 1].set_title('train X2 with Y')

axs[0, 2].plot(X_train.T[2], Y_train)
axs[0, 2].set_title('train X3 with Y')

axs[1, 0].plot(X_train.T[3], Y_train)
axs[1, 0].set_title('train X4 with Y')

axs[1, 1].plot(X_train.T[4], Y_train)
axs[1, 1].set_title('train X5 with Y')

axs[1, 2].plot(X_train.T[5], Y_train)
axs[1, 2].set_title('train X6 with Y')
# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# Graficar en cada subgráfico
axs[0, 0].plot(X_test.T[0], Y_test)
axs[0, 0].set_title('test X1 with Y')

axs[0, 1].plot(X_test.T[1], Y_test)
axs[0, 1].set_title('test X2 with Y')

axs[0, 2].plot(X_test.T[2], Y_test)
axs[0, 2].set_title('test X3 with Y')

axs[1, 0].plot(X_test.T[3], Y_test)
axs[1, 0].set_title('test X4 with Y')

axs[1, 1].plot(X_test.T[4], Y_test)
axs[1, 1].set_title('test X5 with Y')

axs[1, 2].plot(X_test.T[5], Y_test)
axs[1, 2].set_title('test X6 with Y')
# Ajustar espaciado entre subgráficos
plt.tight_layout()

# Mostrar la gráfica
plt.show()
