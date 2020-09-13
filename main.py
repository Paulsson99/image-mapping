from PIL import Image
import numpy as np

import pygame

from image_transformations.transforms import möbius
from image_transformations.transform import transform_pixel_positions, lerp_image
from image_transformations.utils import lerp

from utils.debug import timer
from utils.image_utils import load_image, save_image_sequence

IMAGE_FILE = 'images/py_logo.png'
OUTPUT_FILE = 'images/transformed.gif'

class Transform_Image:

	def __init__(self, im):
		self.running = False

		self.clock = pygame.time.Clock()
		
		# Image
		self.im = im
		self.bg_image = pygame.surfarray.make_surface(im)
		self.transformed_images = []
		self.grid = None
		self.transformed_grid = None

		# Window
		self.window = None
		self.window_size = im.shape[:2]

		# Choose transform settings
		self.is_placing_markers = True
		self.active_marker = 0
		self.marker_pos = []
		self.marker_radius = 10
		self.marker_colors = [
			# Original		Transform
			(255, 0, 0), (200, 0, 0),
			(0, 255, 0), (0, 200, 0),
			(0, 0, 255), (0, 0, 200)]

		# Animation
		self.frames = 240
		self.active_frame = 0
		self.animation_frames = []
		self.fps = 60



	def start_app(self):
		pygame.init()
		self.window = pygame.display.set_mode(self.window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
		self.running = True

	def event_handler(self, event):
		# Quit events
		if event.type == pygame.QUIT:
			self.running = False
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				self.running = False

		# Enter event
			elif event.key == pygame.K_RETURN:
				if self.is_placing_markers:
					if self.active_marker >= 6:
						self.is_placing_markers = False
						self.calc_transform()
					else:
						print('Only {} markers placed. To fully define a transform 6 are needed.'.format(len(self.marker_pos) - 1))

		# Save event
			elif event.key == pygame.K_s:
				self.save()

		# Click events
		elif event.type == pygame.MOUSEBUTTONUP:
			# Left click
			if event.button == 1:
				if self.is_placing_markers:
					if self.active_marker >= 6:
						if (marker := self.clicked_on_marker()) is not None:
							self.active_marker = marker
					else:
						self.place_marker()


	def update(self):
		if self.is_placing_markers:
			self.update_marker()
		else:
			self.update_animation()

	def update_animation(self):
		if len(self.transformed_images) < self.frames:
			t = self.active_frame / (self.frames - 1)
			im = lerp_image(self.grid, self.transformed_grid, self.im, t)
			self.transformed_images.append(im)

			self.animation_frames.append(pygame.surfarray.make_surface(im))

	def update_marker(self):
		# Cant have more than 6 markers
		if not self.active_marker < 6:
			return

		# Append if active marker is not in the list of markers
		if len(self.marker_pos) == self.active_marker:
			self.marker_pos.append(np.array(pygame.mouse.get_pos()))
		# Update the active marker
		else:
			self.marker_pos[self.active_marker] = np.array(pygame.mouse.get_pos())

	def render(self):
		if self.is_placing_markers:
			self.render_settings()
		else:
			self.render_animation()
		pygame.display.flip()

	def render_settings(self):
		self.window.blit(self.bg_image, (0, 0))
		self.render_markers()

	def render_animation(self):
		self.window.blit(self.animation_frames[self.active_frame], (0, 0))
		self.active_frame = (self.active_frame + 1) % self.frames

	def render_markers(self, markers=None):
		if markers is None:
			markers = self.marker_pos
		for marker, col in zip(markers, self.marker_colors):
			pygame.draw.circle(self.window, col, marker, self.marker_radius)

	def loop(self):
		while self.running:
			for event in pygame.event.get():
				self.event_handler(event)

			self.update()
			self.render()

			self.clock.tick(60)

		self.quit_event()

	def quit_event(self):
		pygame.quit()

	def place_marker(self):
		# Place a marker and increase the index for the active marker
		self.marker_pos[self.active_marker] = np.array(pygame.mouse.get_pos())
		self.active_marker = len(self.marker_pos)

	def clicked_on_marker(self):
		mouse = np.array(pygame.mouse.get_pos())
		for i, marker in enumerate(self.marker_pos):
			if np.sum((marker - mouse)**2) < self.marker_radius**2:
				return i
		return None

	def calc_transform(self):
		complex_coords = [x + y * 1j for x, y in self.marker_pos]
		self.grid, self.transformed_grid = transform_pixel_positions(self.im, möbius(*complex_coords))

	def save(self):
		if not len(self.transformed_images) == self.frames:
			print('Only {}/{} images processed'.format(len(self.transformed_images), self.frames))
			return
		print('Saving image to {}'.format(OUTPUT_FILE))
		save_image_sequence(self.transformed_images, OUTPUT_FILE, 1000 / self.fps)



def main():
	# Get image as a numpy array
	im = load_image(IMAGE_FILE)

	TIM = Transform_Image(im)
	TIM.start_app()
	TIM.loop()


	# # Choose the transform
	# coords = choose_transform(im)

	# if not len(coords) == 6:
	# 	print('Quiting early because not enough points were provided.')
	# 	quit()

	# coords = np.array([x + y * 1j for x, y in coords])

	# # Transform the image
	# #transform_func = mirror((1, 0), (0, im.shape[1] / 2))
	# transform_func = möbius(*coords)
	# #transform_func = mirror((1, 1), (im.shape[0] / 2, im.shape[1] / 2))
	# print('Calculation transformation...')
	# im_sequence = transform_image_to_sequence(im, transform_func, FRAME_COUNT)

	# # Save
	# print('Saving image to {}'.format(OUTPUT_FILE))
	# save_image_sequence(im_sequence, OUTPUT_FILE, DURATION)
	# print('Done')

	# # Convert a numpy array into an image object
	# pil_img = Image.fromarray(im_sequence[-1])
	# # Show image
	# pil_img.show()


if __name__ == '__main__':
	main()

