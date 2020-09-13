import numpy as np

def get_grid(x_size, y_size):
	x = np.arange(x_size)
	y = np.arange(y_size)
	xx, yy = np.meshgrid(x, y)
	return np.column_stack((xx.flatten(), yy.flatten()))

def valid_coordinates(grid, min_x, max_x, min_y, max_y):
	x_coord_in_range = np.logical_and(min_x <= grid[...,0], grid[...,0] < max_x)
	y_coord_in_range = np.logical_and(min_y <= grid[...,1], grid[...,1] < max_y)
	return np.logical_and(x_coord_in_range, y_coord_in_range)

def lerp(grid0, grid1, t):
	return grid0 + (grid1 - grid0) * t

