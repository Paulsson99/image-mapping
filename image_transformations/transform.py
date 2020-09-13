import numpy as np

from utils.debug import timer
from .utils import *

@timer
def transform_image_to_sequence(im, f, frames):
	y_size, x_size, color_size = im.shape

	# Create a grid of points for every pixel in the image
	grid = get_grid(x_size, y_size)

	# Transform the grid
	transformed_grid = f(grid)
	
	im_sequence = np.zeros((frames, y_size, x_size, color_size), dtype=im.dtype)
	for frame in range(frames):
		# Linear interpolation between grid and transformed grid
		t = frame / (frames - 1) if frames > 1 else 1
		
		# Set the new pixel colors
		im_sequence[frame,...] = lerp_image(grid, transformed_grid, im, t)

	return im_sequence

def lerp_image(grid1, grid2, im, t):
	y_size, x_size, color_size = im.shape

	# Linear interpolation between grid1 and grid2
	grid = lerp(grid1, grid2, t)

	# Round to closest pixel coordinate
	grid = np.round(grid).astype(int)

	# Mask for all corinates within the image border
	coord_in_range = valid_coordinates(grid, 0, x_size, 0, y_size)

	valid_lerp = grid[coord_in_range,:]
	valid_grid = grid1[coord_in_range,:]

	lerp_im = np.zeros((y_size, x_size, color_size), dtype=im.dtype)
	lerp_im[valid_lerp[:,1], valid_lerp[:,0],:] = im[valid_grid[:,1], valid_grid[:,0],:]

	return lerp_im

def transform_pixel_positions(im, f):
	y_size, x_size, color_size = im.shape

	# Create a grid of points for every pixel in the image
	grid = get_grid(x_size, y_size)

	# Transform the grid
	transformed_grid = f(grid)

	return (grid, transformed_grid)


