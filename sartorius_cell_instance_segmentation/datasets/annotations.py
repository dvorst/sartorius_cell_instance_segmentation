import torch
import pandas as pd
import torchvision
import torchvision.transforms.functional


class Annotations:
	def __init__(self, img_data: pd.DataFrame, idx: str, dtype):
		"""
		Obtain annotations and remove overlap

		Wherever two annotations overlap, each overlapping pixel will be
		assigned to the closest annotation and removed from the other.
		"""

		# decompress annotations and correct the overlapping pixels
		self._annotations = self._correct_overlap(
			self._decompress(img_data, idx, dtype), dtype
		)

		# determine boundaries of each annotation
		bounds_tensor, self.bounds = self._bounds(self._annotations)

		# determine boundaries that touch of neighbouring annotations
		touch_tensor, self.touch = self._touch(bounds_tensor)

		# calculate mask from the annotations and touching boundaries
		self.mask = self._mask(self._annotations, touch_tensor)

	def __call__(self, *args, **kwargs):
		return self._annotations

	@classmethod
	def _correct_overlap(cls, annotations, dtype):
		# calculate center of each annotation
		#     after pixels are added/removed, the center changes, the choice
		#     is made not to recalculate the center in such case due to the
		#     computational overhead involved
		# todo: find a more computational efficient way of implementing this
		centers = torch.stack([cls._center(a, dtype) for a in annotations])

		# find all overlapping pixels
		overlap = cls._overlap(annotations)

		# locations of overlapping pixels
		loc = torch.nonzero(overlap, as_tuple=False)

		# if there are no overlapping pixels, nothing has to be done
		if len(loc) == 0:
			return annotations

		# create meshgrid of loc and centers,
		#    this allows for faster computations than for loops
		loc_mg = loc.repeat(centers.shape[0], 1, 1)
		centers_mg = centers.repeat(loc.shape[0], 1, 1).transpose(0, 1)

		# calculate distances between each overlapping pixel and
		# annotation-center. Taking the root is omitted for speed
		d = torch.sum((loc_mg - centers_mg) ** 2, 2)

		# map the overlapping pixel to the annotation it lies closest to
		map_ = torch.argmin(d, 0)

		# change the annotations accordingly
		# todo: find a faster method for this
		for idx_pixel, idx_annotation in enumerate(map_):
			x, y = loc[idx_pixel]
			# set pixel x, y to zero for all annotations except the
			# annotation that the particular pixel belongs to
			annotations[:, x, y] = 0
			annotations[idx_annotation, x, y] = 1

		return annotations

	@classmethod
	def _decompress(cls, data, idx, dtype):
		return torch.stack([
			cls._decompress_single(
				r['annotation'], r['width'], r['height'], dtype
			)
			for _, r in data.loc[data['id'] == idx].iterrows()
		])

	@staticmethod
	def _decompress_single(annotation_str: str, width, height, dtype):
		# convert annotation string to matrix:
		# [column 1: pixel index, column 2: length]
		annotation_tensor = torch.tensor(
			[int(elm) for elm in annotation_str.split(' ')]
		).reshape([-1, 2])

		# create empty 1D image
		annotation_img_flat = torch.zeros([height * width], dtype=dtype)

		# draw white pixels in the image as indicated by the annotation
		for idx, length in annotation_tensor:
			annotation_img_flat[idx:(idx + length)] = 1.

		# reshape image to 2D
		annotation_img = annotation_img_flat.reshape([height, width])

		# return image
		return annotation_img

	@staticmethod
	def _center(annotation, dtype):
		return torch.round(torch.mean(torch.nonzero(
			annotation, as_tuple=False
		).type(dtype), 0))

	@staticmethod
	def _overlap(annotations):
		"""
		finds all overlapping pixels

		annotations have a pixel value equal to either 0 or 1. So if two or
		more annotations overlap, the sum of these is >1. Overlap is
		returned as torch tensor with values that lie between 0 and 1.
		"""
		return torch.clamp(torch.sum(annotations, 0) - 1, 0, 1)

	@staticmethod
	def _to_pil(t: torch.tensor):
		""" transform tensor with values between 0 and 1 to PIL image """
		return torchvision.transforms.functional.to_pil_image(
			(t * 255).type(torch.uint8)
		)

	@staticmethod
	def from_mask(mask):
		""" exports the provided mask to the challenges desired format. """
		pass

	@staticmethod
	def export():
		pass

	@classmethod
	def _bounds(cls, annotations):
		""" Image of touching boundaries from neighbouring annotations """

		# efficiently calculate gradient in x and y direction, knowing that
		#   all values and result are/should be boolean (i.e. 1's and 0's).
		n, y, x = annotations.shape
		dx = torch.zeros(n, y, x + 2)
		dy = torch.zeros(n, y + 2, x)
		dx[:, :, 1:-1] = annotations
		dy[:, 1:-1, :] = annotations
		dx = torch.abs(dx[:, :, 2:] - dx[:, :, :-2])
		dy = torch.abs(dy[:, 2:, :] - dy[:, :-2, :])
		bounds = dx + dy
		bounds[bounds > 1] = 1
		bounds = torch.sum(bounds, 0)

		# return bounds as tensor and as image
		return bounds, cls._to_pil(bounds.clamp(0, 1))

	@classmethod
	def _touch(cls, bounds):
		"""
		create touching bounds mask/img

		bounds tensor is created by adding all edges together. The pixel
		value of an edge is 1, so the sum of touching bounds will be >1.
		Hence all pixels with value larger than 1 are boundaries that touch
		"""

		touch = torch.clamp(bounds - 1, 0, 1)

		# return the bounds as pil image
		return touch, cls._to_pil(touch)

	@classmethod
	def _mask(cls, annotations, touch):
		"""
		create mask from annotations and bounds (i.e. touching boundaries
		between annotations).

		any touching boundaries between annotations will be considered
		'background' such that the annotations remain semantically
		segmented. The mask is boolean, i.e. the pixel values are either 0
		or 1.
		"""

		# calculate mask
		mask = torch.clamp(
			torch.sum(annotations, 0) - touch,
			0, 1
		)

		# return mask as PIL image
		return cls._to_pil(mask)
