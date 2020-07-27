import pandas as pd
from numpy import save
import numpy as np
data = pd.read_csv("5289_pos_PCPseDNC_training_code_normalized_4000.txt", header=None)
data1 = pd.read_csv("5289_pos_PCPseDNC_training_code_normalized_8000.txt", header=None)
data2 = pd.read_csv("5289_pos_PCPseDNC_training_code_normalized_12000.txt", header=None)
del data[0]
del data1[0]
del data2[0]

a = data.to_numpy()
b = data1.to_numpy()
c = data2.to_numpy()

d = np.concatenate((a, b, c))
print(d.shape)
# del data[0]
# print(data.shape)
# print(data1.shape)
# print(data2.shape)

save('A.thaliana5289_pos_PCPseDNC_4000.npy', d)
