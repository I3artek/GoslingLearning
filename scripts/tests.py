import numpy as np


dupa = np.ones((2,2))

dupa = np.pad(dupa, (0,6), mode='constant')
print(dupa)