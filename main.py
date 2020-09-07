from PIL import Image
import numpy as np

from transforms import *

IMAGE_FILE = 'images/py_logo.png'
OUTPUT_FILE = 'images/transformed.png'

def load_image(filename):
	# Load the image as a numpy array
	return np.array(Image.open(filename).convert('RGBA'))

def save_image(im, filename):
	# Save the image
	im.save(filename)

def transform_image(im, f):
	im_shape = im.shape
	im_transform = np.zeros(im_shape, dtype=im.dtype)

	for x in range(im_shape[0]):
		for y in range(im_shape[1]):
			xx, yy = f(x, y)
			xx, yy = int(np.round(xx)), int(np.round(yy))
			if 0 <= xx < im_shape[0] and 0 <= yy < im_shape[1]:
				im_transform[xx, yy, :] = im[x, y, :]

	return im_transform



def main():
	# Get image as a numpy array
	im = load_image(IMAGE_FILE)

	# Transform the image
	im_transform = transform_image(im, mirror((1, 1), (im.shape[0] / 2, im.shape[1] / 2)))

	print(im_transform.shape)

	# Convert a numpy array into an image object
	pil_img = Image.fromarray(im_transform)

	# Save
	save_image(pil_img, OUTPUT_FILE)

	# Show image
	pil_img.show()


if __name__ == '__main__':
	main()

