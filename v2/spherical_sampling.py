from misc import spherical2cartesian

import matplotlib.pyplot as plt
import torch

torch.manual_seed(42)

tp = torch.rand(1000, 2)

tp.T[0] = tp.T[0]*torch.pi/2
tp.T[1] = tp.T[1]*torch.pi*2

uni = torch.rand(1000, 2)
uni.T[0] = torch.arccos(uni.T[0])
uni.T[1] = uni.T[1]*torch.pi*2

plt.figure(figsize=[10, 5])
plt.subplot(1, 2, 1)
plt.scatter(*spherical2cartesian(tp))
plt.subplot(1, 2, 2)
plt.scatter(*spherical2cartesian(uni))
plt.show()
