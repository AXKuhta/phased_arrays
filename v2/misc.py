import torch

def spherical2cartesian(directions):
	t, p = directions.T

	return torch.vstack([
		t.sin() * p.cos(),
		t.sin() * p.sin(),
		t.cos()
	])

