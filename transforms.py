import numpy as np

# TODO: Decorator?



def mirror(line, origin):
	line = np.array(line, dtype=np.float64)
	origin = np.array(origin, dtype=np.float64)
	def mirror_func(x, y):
		p = np.array((x, y), dtype=np.float64)
		p -= origin
		p = 2 * np.dot(line, p) / np.dot(line, line) * line - p
		return p + origin
	return mirror_func