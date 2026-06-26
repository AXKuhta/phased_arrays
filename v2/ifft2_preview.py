import matplotlib.pyplot as plt
import numpy as np
import torch

from misc import spherical2cartesian

f = 100*1000*1000
c = 299792458
l = c/f
d = l/2 # element spacing

kd = 2*np.pi*d/l

# Element locations (virtual)
m, n = np.meshgrid(np.arange(10), np.arange(10))
m = m.flatten()
n = n.flatten()
z = np.zeros_like(m)

pts = torch.tensor(np.float32(np.dstack([m, n, z])[0]))

# Tapering (optional)
taper = torch.tensor(np.hamming(10))
taper = torch.outer(taper, taper).flatten()

v = spherical2cartesian(torch.deg2rad(torch.tensor([0, 0])))

w = torch.exp(pts @ v * 1j * kd).T * taper
w = w.reshape(-1, 10, 10)

af = torch.fft.ifft2(w, [500, 500])*500*500
af = torch.fft.fftshift(af, [-2, -1])

plt.imshow(af[0].abs(), norm="log")
plt.show()
