from PIL import Image
import numpy as np

from transforms import *

from debug import timer

IMAGE_FILE = 'images/py_logo.png'
OUTPUT_FILE = 'images/transformed.gif'

FRAME_COUNT = 100
DURATION = 40 # ms

BACKGROUND = np.array((0, 0, 0))

def load_image(filename):
	# Load the image as a numpy array (saved as [y, x, RGBA])
	return np.array(Image.open(filename).convert('RGBA'))

def save_image_sequence(im_sequence, filename):
	images = []
	for im in im_sequence:
		images.append(Image.fromarray(blend_background(im, BACKGROUND)))
	images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=DURATION, loop=1)

def get_grid(x_size, y_size):
	x = np.arange(x_size)
	y = np.arange(y_size)
	xx, yy = np.meshgrid(x, y)
	return np.column_stack((xx.flatten(), yy.flatten()))
	#return np.array([(x, y) for x in range(x_size) for y in range(y_size)], dtype=int)

@timer
def transform_image(im, f):
	y_size, x_size, color_size = im.shape

	# Create a grid of points for every pixel in the image
	grid = get_grid(x_size, y_size)

	# Transform the grid
	transformed_grid = f(grid)
	
	im_sequence = np.zeros((FRAME_COUNT, y_size, x_size, color_size), dtype=im.dtype)
	for frame in range(FRAME_COUNT):
		# Linear interpolation between grid and transformed grid
		t = frame / (FRAME_COUNT - 1) if FRAME_COUNT > 1 else 1
		frame_grid = lerp(grid, transformed_grid, t)

		# Round to closest pixel coordinate
		frame_grid = np.round(frame_grid).astype(int)

		# Mask for all corinates within the image border
		coord_in_range = valid_coordinates(frame_grid, 0, x_size, 0, y_size)

		valid_transform = frame_grid[coord_in_range,:]
		valid_grid = grid[coord_in_range,:]

		# Set the new pixel colors
		im_sequence[frame, valid_transform[:,1], valid_transform[:,0],:] = im[valid_grid[:,1], valid_grid[:,0], :]

	return im_sequence

def transform_grid(grid, f):
	# Transform a grid with the function f and round to closest pixel
	return f(grid)

def valid_coordinates(grid, min_x, max_x, min_y, max_y):
	x_coord_in_range = np.logical_and(min_x <= grid[...,0], grid[...,0] < max_x)
	y_coord_in_range = np.logical_and(min_y <= grid[...,1], grid[...,1] < max_y)
	return np.logical_and(x_coord_in_range, y_coord_in_range)

def lerp(grid0, grid1, t):
	return grid0 + (grid1 - grid0) * t

def blend_background(colors, bg):
	alpha = (colors[...,3] / 255)[...,np.newaxis]
	new_color = colors[...,:3] * alpha + (1 - alpha) * bg
	return np.round(new_color).astype(np.uint8)





def main():
	# Get image as a numpy array
	im = load_image(IMAGE_FILE)

	# Transform the image
	#transform_func = mirror((1, 0), (0, im.shape[1] / 2))
	transform_func = möbius(0, 512, 512, 512+512j, 512+512j, 512j)
	#transform_func = mirror((1, 1), (im.shape[0] / 2, im.shape[1] / 2))
	print('Calculation transformation...')
	im_sequence = transform_image(im, transform_func)

	# Save
	print('Saving image to {}'.format(OUTPUT_FILE))
	save_image_sequence(im_sequence, OUTPUT_FILE)
	print('Done')

	# Convert a numpy array into an image object
	pil_img = Image.fromarray(im_sequence[-1])
	# Show image
	pil_img.show()


if __name__ == '__main__':
	main()

