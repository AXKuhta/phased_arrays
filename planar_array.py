import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)

#
# https://en.wikipedia.org/wiki/Spherical_coordinate_system#Coordinate_system_conversions
#
# phi		azimuth
# theta		inclination (as opposed to elevation)
#

def beam(phi, theta):
	vec = np.array([
		np.sin(theta)*np.cos(phi),
		np.sin(theta)*np.sin(phi),
		np.cos(theta)
	])

	m, n = np.meshgrid(np.arange(4), np.arange(4))
	m = m.flatten()
	n = n.flatten()
	z = np.zeros_like(m)

	pts = np.dstack([m, n, z])[0]

	lag = pts @ vec

	f = 100*1000*1000
	c = 299792458
	l = c/f
	d = l/2 # element spacing

	kd = 2*np.pi*d/l

	radiators = np.exp(1j * lag * kd)

	return np.sum(radiators)

a = np.linspace(-180, +180, 360) / 57.2
b = np.linspace(-180, +180, 360) / 57.2

pts = []

for theta in a:
	for phi in b:
		m = beam(phi, theta)

		vec = np.array([
			np.sin(theta)*np.cos(phi),
			np.sin(theta)*np.sin(phi),
			np.cos(theta)
		]) * np.abs(m)

		pts.append(vec)

pts = np.array(pts)

for x, y, z in pts:
	print(f"v {x} {y} {z}")
