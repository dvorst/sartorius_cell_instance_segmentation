import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional
import PIL.Image
import matplotlib.pyplot as plt
import pathlib
import zipfile
import cv2
import dataclasses

from .annotations import to_annotation
from sartorius_cell_instance_segmentation.util import imshow


class SupervisedDataset(torch.utils.data.Dataset):
	@dataclasses.dataclass
	class Item:
		img: torch.Tensor
		canny: torch.Tensor
		bounds: torch.Tensor
		touch: torch.Tensor
		mask: torch.Tensor

	def __init__(
			self,
			transforms=None,
			target_transforms=None,
			dtype=torch.float,
			train_csv: str = '../input/sartorius-cell-instance-segmentation/train.csv',
			dir_imgs: str = '../input/sartorius-cell-instance-segmentation/train',
			zip_data: str = 'supervisedData.zip',
			imgs_extension: str = 'png',
			force_convert=False,
			print_progress=True,
			n_imgs=None,
	):
		self.transforms = transforms
		self.target_transforms = target_transforms
		self.dtype = dtype

		# remove zip file when forced to re-convert the data
		if force_convert:
			if pathlib.Path(zip_data).exists():
				pathlib.Path(zip_data).unlink()

		# convert data to zip if not done already
		if not pathlib.Path(zip_data).exists():
			self._convert_data(train_csv, dir_imgs, zip_data, imgs_extension, print_progress, n_imgs)

		# load data
		self.items, self.map = self._load_data(
			zip_data, imgs_extension, n_imgs, dtype
		)

	# sanity check that every img, mask exist in map
	# self._sanity_check(data_ids=self.ids_map, img_ids=self.pths.keys())

	def __len__(self):
		return len(self.map)

	def __getitem__(self, idx):
		item = self.items[self.map[idx]]
		img = item.img
		canny = item.canny
		bounds = item.bounds
		touch = item.touch
		mask = item.mask
		if self.transforms is not None:
			img = self.transforms(img)
		if self.target_transforms is not None:
			bounds = self.target_transforms(bounds)
			canny = self.target_transforms(canny)
			touch = self.target_transforms(touch)
			mask = self.target_transforms(mask)
		return img, canny, bounds, touch, mask

	@classmethod
	def overlay(
			cls, img, bounds, touch, mask, alpha=(0.5, 0.2, 0.2, 0.1),
			col_img=(1., 1., 1.), col_bounds=(0., 0., 1.),
			col_touch=(0., 1., 0.), col_mask=(1., 0., 0.),
	):
		""" overlay mask on image """

		# verify that sum of all three alpha channels is 1
		if not (0.999999999999 < sum(alpha) < 1.000000000001):
			raise ValueError(f'sum of alpha must be 1 but is {sum(alpha)}')

		# convert img/mask to colored images
		img = cls._gray_img_to_col(img, col_img)
		bounds = cls._gray_img_to_col(bounds, col_bounds)
		touch = cls._gray_img_to_col(touch, col_touch)
		mask = cls._gray_img_to_col(mask, col_mask)
		# return as PIL image
		return cls.tensor_to_pil(
			img * alpha[0] +
			bounds * alpha[1] +
			touch * alpha[2] +
			mask * alpha[3]
		)

	@staticmethod
	def tensor_to_pil(t: torch.tensor):
		""" transform tensor with values between 0 and 1 to PIL image """
		return torchvision.transforms.functional.to_pil_image(
			(t * 255).type(torch.uint8)
		)

	@classmethod
	def _convert_data(
			cls, train_csv, dir_imgs, zip_data, imgs_extension, print_progress, n_imgs):
		""" Convert and save data provided by the challenge as a zipfile """

		print('SupervisedDataset: converting data...')

		# load train.csv file as pandas dataframe
		data = pd.read_csv(train_csv)

		# get paths of imgs located in dir_imgs
		pths = list(pathlib.Path(dir_imgs).glob(f'*.{imgs_extension}'))

		# create data zip
		with zipfile.ZipFile(zip_data, 'w') as zf:
			# loop through pths
			for idx, pth in enumerate(pths):
				if n_imgs is not None and idx == n_imgs:
					break

				# the imgs, masks and bounds have the same filename
				filename = f'{pth.stem}.{imgs_extension}'

				# convert data to annotations
				# annotations = Annotations(data, pth.stem)
				annotations = to_annotation(data, pth.stem)

				# add img to zipfile
				zf.write(pth, f'imgs/{filename}')

				# Add the annotation boundaries
				with zf.open(f'bounds/{filename}', 'w') as file:
					annotations.bounds.save(file, 'png', optimize=True)

				# Add the annotation boundaries that touch each other
				with zf.open(f'touch/{filename}', 'w') as file:
					annotations.touch.save(file, 'png', optimize=True)

				# add the annotation mask
				with zf.open(f'masks/{filename}', 'w') as file:
					annotations.mask.save(file, 'png', optimize=True)

				# print progress
				print(
					f'SupervisedDataset: \t{idx + 1}/{len(pths)} converted '
					f'{str(pth)}'
				)

	@classmethod
	def _load_data(cls, zip_data, imgs_extension, n_imgs, dtype):
		""" load zip file data """

		# open zipfile
		with zipfile.ZipFile(zip_data, 'r') as zf:
			# extract filenames to create map = idx_str[idx_int]
			map_ = list(np.sort(np.unique(
				[pathlib.Path(file).stem for file in zf.namelist()]
			)))

			if n_imgs is not None:
				map_ = map_[:n_imgs]

			# read imgs, bounds and masks as dict
			items = {}
			for idx in map_:
				img = cls._load_zip_img(zf, f'imgs/{idx}.{imgs_extension}', dtype)
				canny = cls._canny(img, dtype)
				bounds = cls._load_zip_img(zf, f'bounds/{idx}.{imgs_extension}', dtype)
				touch = cls._load_zip_img(zf, f'touch/{idx}.{imgs_extension}', dtype)
				mask = cls._load_zip_img(zf, f'masks/{idx}.{imgs_extension}', dtype)
				items[idx] = cls.Item(img=img, canny=canny, bounds=bounds, touch=touch, mask=mask)

		return items, map_

	@staticmethod
	def _canny(img, dtype):
		return torch.tensor(cv2.Canny(
			(img.numpy() * 255).astype(np.uint8).squeeze(), threshold1=10, threshold2=100
		) / 255., dtype=dtype).unsqueeze(0)

	@staticmethod
	def _gray_img_to_col(gray_img: torch.tensor, color):
		return torch.cat([gray_img * c for c in color], dim=0)

	@staticmethod
	def _load_zip_img(zf, filename, dtype):
		with zf.open(filename, 'r') as file:
			return (torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(file)) / 255).to(dtype)
			# return torch.as_tensor(np.array(
			#     PIL.Image.open(file)), dtype=dtype
			# ).unsqueeze(0) / 255  # change dtype to inherit dtype of class

	@staticmethod
	def _sanity_check(data_ids, img_ids):
		# todo: needs to be extended to include map
		for data_id in data_ids:
			if data_id not in img_ids:
				raise Exception(f'no img found for id: {data_id}')
