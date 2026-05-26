import matplotlib.pyplot as plt
import numpy as np
import torch

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

	pts = np.dstack([m, n, z])[0]

	# Pre-steering
	t = 10/57.2
	p = 10/57.2
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
	w = torch.exp(torch.tensor(pts) @ v * 1j * kd) * taper
	w.requires_grad = True

	#
	# phi           azimuth
	# theta         inclination (as opposed to elevation)
	#
	def array_factor(self, phi, theta):
		pts = self.pts
		kd = self.kd

		vec = np.array([
			np.sin(theta)*np.cos(phi),
			np.sin(theta)*np.sin(phi),
			np.cos(theta)
		])

		vec = torch.tensor(vec + .0)
		pts = torch.tensor(pts + .0)

		# Lag distance in a direction (virtual, to a far field point)
		lag = pts @ vec

		radiators = self.w[:, None] * torch.exp(1j * lag * kd)

		return torch.sum(radiators, 0)

	def loss(self, af, mask_upper, mask_lower):
		af = torch.abs(af)
		u_loss = torch.linalg.norm(torch.maximum(torch.zeros_like(af), af - mask_upper))
		l_loss = torch.linalg.norm(torch.minimum(torch.zeros_like(af), af - mask_lower))

		return u_loss + l_loss

model = PlanarArray()

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
mainlobe = spherical_to_vector(np.radians(10), np.radians(10))
cos_angs = vectors @ mainlobe
angs = np.degrees(np.arccos(cos_angs))
mask_mainlobe = angs < 5

mask_upper = 90*mask_mainlobe + 10
mask_lower = 100*mask_mainlobe

grid = torch.tensor(grid)
mask_upper = torch.tensor(mask_upper)
mask_lower = torch.tensor(mask_lower)

def picture_raw(phi, theta, af):
	plt.pcolormesh(a, b, np.abs(af).reshape(90, 360))
	plt.contour(a, b, np.abs(af).reshape(90, 360)>0.5*np.max(af, 0), levels=[0.5], colors='red', linewidths=2)
	plt.gca().set_aspect("equal")
	plt.ylabel("Inclination")
	plt.xlabel("Azimuth")
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
	plt.show()

af = model.array_factor(*grid.T)
#picture_raw(a, b, af.detach().numpy())
picture(*grid.T.numpy(), af.detach().numpy())

picture(*grid.T.numpy(), mask_mainlobe)
#picture_raw(a, b, mask_mainlobe)

for i in range(100):
	af = model.array_factor(*grid.T)

	optim = torch.optim.SGD([model.w])
	#optim = torch.optim.Adam([model.w])
	loss = model.loss(af, mask_upper, mask_lower)

	optim.zero_grad()
	loss.backward()
	optim.step()

	print(loss)

picture(*grid.T.numpy(), af.detach().numpy())
