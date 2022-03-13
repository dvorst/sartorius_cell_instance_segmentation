import matplotlib.pyplot as plt
import PIL.Image
import datetime
import torch
import torchvision
import torchvision.transforms.functional
from typing import Union


def tensor_to_pil(t: torch.tensor):
	""" transform tensor with values between 0 and 1 to PIL image """
	# verify min/max of tensor
	min_, max_ = torch.min(t), torch.max(t)
	if max_ > 1:
		raise ValueError('max(tensor) = %.2f, which is larger than 1' % max_)
	if min_ < 0:
		raise ValueError('min(tensor) = %.2f, which is smaller than 0' % min_)
	# return pil image
	return torchvision.transforms.functional.to_pil_image(
		(t * 255).type(torch.uint8)
	)


def imshow(img: Union[type(PIL.Image), torch.Tensor]):
	with torch.no_grad():
		# convert to pil image if tensor
		img = tensor_to_pil(img.detach()) if isinstance(img, torch.Tensor) else img

		# determine whether image is gray or colored
		ch = 1 if isinstance(img.getpixel((0, 0)), int) else 3
		cmap = 'gray' if ch == 1 else plt.rcParams['image.cmap']

		# plot img without axis and tight layout
		plt.figure(dpi=200)
		plt.imshow(img, cmap=cmap)
		plt.gca().set_axis_off()
		plt.tight_layout()


def timestamp(for_file: bool = False):
	""" returns current date and time, can also be used for filenames if
	for_file is set to true"""
	if for_file:
		return datetime.datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
	else:
		return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


