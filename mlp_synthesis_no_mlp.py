from torch.optim.lr_scheduler import LinearLR
from time import perf_counter
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

import scipy

#
# Reproduction attempt for a questionable paper
# DOI: 10.1049/mia2.12290
#

class PlanarArray(torch.nn.Module):
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

	pts = torch.tensor(np.dstack([m, n, z])[0] + 0.0)

	# Pre-steering
	t = 45/57.2
	p = 45/57.2
	v = -np.array([
		np.sin(t)*np.cos(p),
		np.sin(t)*np.sin(p),
		np.cos(t)
	])

	# Tapering (optional)
	taper = torch.tensor(np.hamming(10))
	taper = torch.outer(taper, taper).flatten()

	# Array weights
	# Should be all 1s for beam up
	# Warning: ensure the datatype is complex
	w = torch.exp(pts @ v * 1j * kd) * taper
	w.requires_grad = True

	from torch.nn import Linear, ReLU, Tanh

	inputs = torch.tensor([0.0, 0.0])

	act = ReLU()
	fin = Tanh()
	w1 = Linear(2, 4096)
	w2 = Linear(4096, 2048)
	w3 = Linear(2048, 2048)
	w4 = Linear(2048, 1024)

	re = Linear(1024, 100)
	im = Linear(1024, 100)

	torch.nn.init.xavier_uniform_(w1.weight, gain=torch.nn.init.calculate_gain("relu"))
	torch.nn.init.xavier_uniform_(w2.weight, gain=torch.nn.init.calculate_gain("relu"))
	torch.nn.init.xavier_uniform_(w3.weight, gain=torch.nn.init.calculate_gain("relu"))
	torch.nn.init.xavier_uniform_(w4.weight, gain=torch.nn.init.calculate_gain("relu"))
	torch.nn.init.xavier_uniform_(re.weight, gain=torch.nn.init.calculate_gain("relu"))
	torch.nn.init.xavier_uniform_(im.weight, gain=torch.nn.init.calculate_gain("relu"))

	#
	# phi           azimuth
	# theta         inclination (as opposed to elevation)
	#
	def array_factor(self, phi, theta):
		pts = self.pts
		kd = self.kd

		vec = torch.stack([
			np.sin(theta)*np.cos(phi),
			np.sin(theta)*np.sin(phi),
			np.cos(theta)
		])

		# Lag distance in a direction (virtual, to a far field point)
		lag = pts @ vec

		radiators = self.w[:, None] * torch.exp(1j * lag * kd)

		return torch.sum(radiators, 0)

	def array_factor_fast_init(self, phi, theta):
		pts = self.pts
		kd = self.kd

		vec = torch.stack([
			np.sin(theta)*np.cos(phi),
			np.sin(theta)*np.sin(phi),
			np.cos(theta)
		])

		# Lag distance in a direction (virtual, to a far field point)
		lag = pts @ vec

		self.delta_phi = torch.exp(1j * lag * kd).to(dtype=torch.complex64)

	def array_factor_fast(self):
		radiators = self.w[:, None] * self.delta_phi

		return torch.sum(radiators, 0)

	def array_factor_fast_nn(self):
		x = self.inputs
		x = self.act(self.w1(x))
		x = self.act(self.w2(x))
		x = self.act(self.w3(x))
		x = self.act(self.w4(x))

		w = torch.complex(
			self.fin(self.re(x)),
			self.fin(self.im(x))
		)

		self.w = w

		radiators = w[:, None] * self.delta_phi

		return torch.sum(radiators, 0)

	def loss(self, af, mask_upper, mask_lower):
		af = torch.abs(af)
		z = torch.tensor(0)
		#u_loss = af - mask_upper
		#u_loss = torch.dot(u_loss, u_loss)/90/360
		#l_loss = 0
		ovr = torch.maximum(z, af - mask_upper)
		und = torch.minimum(z, af - mask_lower)
		u_loss = torch.dot(ovr, ovr)/90/360
		l_loss = torch.dot(und, und)/90/360

		return u_loss + l_loss

model = PlanarArray()

model.inputs[0] = np.radians(45)
model.inputs[1] = np.radians(45)

a = np.linspace(-180, +180, 360)
b = np.linspace(0, +90, 90)
u, v = np.meshgrid(a/57.2, b/57.2)
grid = np.dstack([u.flatten(), v.flatten()])[0]

def spherical_to_vector(azi, inc):
	x = np.sin(inc) * np.cos(azi)
	y = np.sin(inc) * np.sin(azi)
	z = np.cos(inc)
	return np.stack([x, y, z]).T

vectors = spherical_to_vector(*grid.T)
mainlobe = spherical_to_vector(np.radians(45), np.radians(45))
cos_angs = vectors @ mainlobe
angs = np.degrees(np.arccos(cos_angs))

mask_upper = 25*(angs<10)
mask_lower = 25*(angs<5)

grid = torch.tensor(grid)
mask_upper = torch.tensor(mask_upper)
mask_lower = torch.tensor(mask_lower)

def picture_raw(phi, theta, af):
	plt.pcolormesh(a, b, np.abs(af).reshape(90, 360))
	#plt.contour(a, b, np.abs(af).reshape(90, 360)>0.5*np.max(af, 0), levels=[0.5], colors='red', linewidths=2)
	#plt.gca().set_aspect("equal")
	plt.ylabel("Inclination")
	plt.xlabel("Azimuth")

def picture_w_sidelobes(u, v, af, show=True):
	lobe_bitmap = scipy.ndimage.laplace(np.abs(af).reshape(90, 360))<0
	features, count = scipy.ndimage.label(lobe_bitmap)

	pk_sort = []
	pk_gain = []
	pk_ind = []

	for i in range(1, count):
		mask = (features.flatten() == i)
		peak = np.max(np.abs(af)*mask)
		ind = np.argmax(np.abs(af)*mask)
		pk_gain.append(peak)
		pk_ind.append(ind)

	ind = np.argsort(pk_gain)
	pk_gain = np.array(pk_gain)[ind]
	pk_ind = np.array(pk_ind)[ind]

	x = np.sin(v)*np.sin(u)
	y = np.sin(v)*np.cos(u)

	max = np.max(np.abs(af))

	half = np.abs(af).reshape(90, 360)>max/np.sqrt(2)
	xset = x[half]
	yset = y[half]

	xmin = np.degrees(np.asin(np.min(xset)))
	xmax = np.degrees(np.asin(np.max(xset)))
	ymin = np.degrees(np.asin(np.min(yset)))
	ymax = np.degrees(np.asin(np.max(yset)))

	# Non precise
	xhpbw = np.abs(xmin - xmax)
	yhpbw = np.abs(ymin - ymax)

	from matplotlib.lines import Line2D

	cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
	color_a = cycler.pop(0)
	color_b = cycler.pop(0)

	plt.pcolormesh(x, y, 20*np.log10(np.abs(af).reshape(90, 360)/max), shading="gouraud")
	cb = plt.colorbar(label="20log10(|AF|)")
	plt.contour(x, y, np.abs(af).reshape(90, 360)>max/np.sqrt(2), levels=[0.5], colors='red', linewidths=2)
	#plt.contour(x, y, lobe_bitmap, levels=[0.5], colors='blue', linewidths=2)
	plt.gca().legend([Line2D([0], [0], color="red", lw=1)], ["$\Theta_{HPBW}=" + f"{yhpbw:.1f}°" + "$"])
	plt.gca().set_aspect("equal")
	plt.title("$|AF|$")
	plt.ylabel("$\sin(\Theta)\cos(\phi)$")
	plt.xlabel("$\sin(\Theta)\sin(\phi)$")

	for i, (gain, ind) in enumerate(zip(pk_gain[-5:], pk_ind[-5:])):
		x_ = x.flatten()[ind]
		y_ = y.flatten()[ind]

		c = color_a if i == 4 else color_b

		plt.scatter(x_, y_, c=c)
		cb.ax.axhline(20*np.log10(gain/max), c=c, lw=3)

		plt.annotate(f"{gain:.1f}",
			xy=(x_, y_),
			xycoords='data',
			color=c,
			xytext=(0, -15),
			textcoords='offset points',
			weight="bold",
			ha="center",
		)

	if show:
		plt.show()

def picture(phi, theta, af):
	vec = np.array([
		np.sin(theta)*np.cos(phi),
		np.sin(theta)*np.sin(phi),
		np.cos(theta)
	])

	y,x,z=vec
	z = np.abs(af)

	plt.hexbin(x,y,z)
	plt.colorbar()
	plt.ylabel("$\sin(\Theta)\cos(\phi)$")
	plt.xlabel("$\sin(\Theta)\sin(\phi)$")

af = model.array_factor(*grid.T)

af_original = torch.clone(af)

w2 = model.w.clone().detach().numpy()

# Weight reset
model.w = torch.ones([100], dtype=torch.complex64, requires_grad=True)

steps = 1000

optim = torch.optim.SGD([model.w], momentum=0.9, lr=1e-3) # higher lr required
sched = LinearLR(optim, 1.0, 0.0, steps)

loss_log = []
time_log = [perf_counter()]

model.array_factor_fast_init(*grid.T)

for i in range(steps):
	start = perf_counter()

	af = model.array_factor_fast()
	#af = model.array_factor_fast_nn()

	elapsed = perf_counter() - start
	print(f"fwd {elapsed*1000:.1f}ms")

	start = perf_counter()
	loss = model.loss(af, mask_upper, mask_lower)

	optim.zero_grad()
	loss.backward()
	optim.step()
	sched.step()

	elapsed = perf_counter() - start
	print(f"bwd {elapsed*1000:.1f}ms")

	print(loss)

	loss_log.append(loss)
	time_log.append(perf_counter())

with open("no_mlp.json", "w") as f:
	json.dump(dict(
		time = time_log,
		loss = [float(x) for x in loss_log]
	), f)

w3 = model.w.clone().detach().numpy()

plt.subplot(2, 3, 1)
plt.imshow(np.abs(w2).reshape(10, 10))
plt.title("original |w|")

plt.subplot(2, 3, 2)
plt.imshow(np.angle(w2).reshape(10, 10))
plt.title("original arg(w)")

plt.subplot(2, 3, 3)
picture_w_sidelobes(u, v, af_original.detach().numpy(), show=False)

plt.subplot(2, 3, 4)
plt.imshow(np.abs(w3).reshape(10, 10))
plt.title("trained |w|")

plt.subplot(2, 3, 5)
plt.imshow(np.angle(w3).reshape(10, 10))
plt.title("trained arg(w)")

plt.subplot(2, 3, 6)
picture_w_sidelobes(u, v, af.detach().numpy(), show=False)

elapsed = max(time_log) - min(time_log)
plt.suptitle(f"Time elapsed {elapsed*1000:.1f}ms")
plt.show()
