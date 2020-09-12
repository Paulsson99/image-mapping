import numpy as np
from debug import timer

# TODO: Decorator?



def mirror(line, origin):
	# Normalize the line
	line = np.array(line, dtype=np.float64) / np.sqrt(np.dot(line, line))
	origin = np.array(origin, dtype=np.float64)
	def mirror_func(grid):
		grid = grid.astype(float)
		grid -= origin
		s = 2 * np.matmul(grid, line)
		grid = np.matmul(s[:,np.newaxis], line[np.newaxis,:]) - grid
		return grid + origin
	return mirror_func


def möbius(z1, w1, z2, w2, z3, w3):
	# Calculate the möbius function h(z) defined by the cross ratio [z, z1, z2, z3]
	h_a, h_b, h_c, h_d = cross_ratio(z1, z2, z3)
	# Calculate the möbius function g(w) defined by the cross ratio [w, w1, w2, w3]
	g_a, g_b, g_c, g_d = cross_ratio(w1, w2, w3)
	# Calculate the composition of g^(-1)(f(z))
	a, b, c, d = möbius_composition(h_a, h_b, h_c, h_d, g_d, -g_b, -g_c, g_a)

	def möbius_func(grid):
		# Convert the grid to complex numbers
		Z = grid[:, 0] + grid[:, 1] * 1j
		# Möbius tansformation
		Z = (a * Z + b) / (c * Z + d)
		# Convert to a grid again
		return np.column_stack((np.real(Z), np.imag(Z)))

	return möbius_func

def cross_ratio(z1, z2, z3):
	a = z2 - z3
	b = -z1 * (z2 - z3)
	c = z2 - z1
	d = -z3 * (z2 - z1)
	return (a, b, c, d)

def möbius_composition(a, b, c, d, e, f, g, h):
	a_ = a * e + c * f
	b_ = b * e + d * f
	c_ = a * g + c * h
	d_ = b * g + d * h
	return (a_, b_, c_, d_)












