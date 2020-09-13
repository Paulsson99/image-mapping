from PIL import Image
import numpy as np

def load_image(filename, bg=(0, 0, 0)):
	# Load the image as a numpy array (saved as [y, x, RGB])
	im = np.array(Image.open(filename).convert('RGBA'))
	# Blend transparent parts with a background
	im = blend_background(im, np.array(bg))
	return im

def save_image_sequence(im_sequence, filename, duration, loop=1):
	images = [Image.fromarray(im) for im in im_sequence]		
	images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=1)

def blend_background(colors, bg):
	alpha = (colors[...,3] / 255)[...,np.newaxis]
	new_color = colors[...,:3] * alpha + (1 - alpha) * bg
	return np.round(new_color).astype(np.uint8)