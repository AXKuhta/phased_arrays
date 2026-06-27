import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import torch

from torch.nn import Sequential, Linear, ReLU, Tanh
from misc import spherical2cartesian, select

import safetensors.torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sequential(
	Linear(3, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 200), Tanh()
).to(dtype=torch.float32)

checkpoints = list(sorted(glob("runs/*/final.safetensors")))
checkpoint = select(checkpoints)

print(f"Loading {checkpoint}...")

safetensors.torch.load_model(model, checkpoint)


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

theta = 0
phi = 0

plt.figure(figsize=[10, 5])

while True:
	v = spherical2cartesian(torch.deg2rad(torch.tensor([theta, phi])))

	w = torch.exp(pts @ v * 1j * kd).T * taper
	w = w.reshape(-1, 10, 10)

	af = torch.fft.ifft2(w, [500, 500])*500*500
	af = torch.fft.fftshift(af, [-2, -1])

	plt.clf()
	plt.subplot(1, 2, 1)
	plt.imshow(af[0].abs(), norm="log")

	preds = model(v.T)
	re, im = preds.split(100, 1)
	w_ = torch.complex(re, im).reshape(-1, 10, 10)

	af_ = torch.fft.ifft2(w_, [500, 500])*500*500
	af_ = torch.fft.fftshift(af_, [-2, -1])

	plt.subplot(1, 2, 2)
	plt.imshow(af_[0].detach().abs(), norm="log")
	plt.pause(.1)

	theta = float(input("Theta>"))
	phi = float(input("Phi>"))
