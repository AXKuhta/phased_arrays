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

	#
	# phi           azimuth
	# theta         inclination (as opposed to elevation)
	#
	def array_factor(self, phi, theta):
		pts = self.pts
		kd = self.kd

		t = 0/57.2
		p = 0/57.2

		vec = np.array([
			np.sin(theta)*np.cos(phi),
			np.sin(theta)*np.sin(phi),
			np.cos(theta)
		]) - np.array([
			np.sin(t)*np.cos(p),
			np.sin(t)*np.sin(p),
			np.cos(t)
		])[:, None]

		# Lag distance in a direction (virtual, to a far field point)
		lag = pts @ vec

		radiators = np.exp(1j * lag * kd)

		return np.sum(radiators, 0)

model = PlanarArray()

a = np.linspace(-180, +180, 360)
b = np.linspace(0, +90, 180)
u, v = np.meshgrid(a/57.2, b/57.2)
grid = np.dstack([u.flatten(), v.flatten()])[0]

af = model.array_factor(*grid.T)

def picture_raw(phi, theta, af):
	plt.pcolormesh(a, b, np.abs(af).reshape(180, 360))
	plt.contour(a, b, np.abs(af).reshape(180, 360)>50, levels=[0.5], colors='red', linewidths=2)
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

#picture_raw(a, b, af)
picture(*grid.T, af)
