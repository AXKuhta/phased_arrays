from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import scipy

from torch.nn import Sequential, Linear, ReLU, Tanh
from misc import spherical2cartesian
from ds import train_dataloader

writer = SummaryWriter()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Sequential(
	Linear(2, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 1024), ReLU(),
	Linear(1024, 200), Tanh()
).to(dtype=torch.float32).to(device)

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

optim = torch.optim.AdamW(model.parameters())
sched = LinearLR(optim, 1.0, 0.0, len(train_dataloader))

# Device upload
taper = taper.to(device)
pts = pts.to(device)

ts = []

def it_per_second():
	now = perf_counter()

	for i, x in enumerate(reversed(ts)):
		if x < now - 1:
			return i + 1

	return 0

for i, directions in enumerate(train_dataloader):
	start = perf_counter()

	#
	# Device download/upload
	#
	directions = directions.to(device)

	v = spherical2cartesian(directions)

	w = torch.exp(pts @ v * 1j * kd).T * taper

	preds = model(directions)
	re, im = preds.split(100, 1)

	loss_re = w.real - re
	loss_im = w.imag - im

	loss = torch.sum(loss_re*loss_re + loss_im*loss_im)

	optim.zero_grad()
	loss.backward()
	optim.step()
	sched.step()

	ts.append(perf_counter())
	elapsed = perf_counter() - start

	itps = it_per_second()

	#print(f"{itps}it/sec {i}/{len(train_dataloader)}\t{loss:.1f}")
	if i % 50 == 0:
		writer.add_scalar("Loss", loss, i)
		writer.add_scalar("Perf/it per sec", itps, i)
		writer.add_scalar("Perf/ms per step", elapsed*1000, i)
		writer.add_scalar("Perf/GPU%", torch.cuda.utilization(), i)

	#af = torch.fft.ifft2(w, [500, 500])*500*500
	#af = torch.fft.fftshift(af, [-2, -1])
