import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional
import PIL
import matplotlib.pyplot as plt
import pathlib
import zipfile

from .annotations import Annotations
from sartorius_cell_instance_segmentation.util import imshow


class SupervisedDataset(torch.utils.data.Dataset):
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

		if force_convert:
			if pathlib.Path(zip_data).exists():
				pathlib.Path(zip_data).unlink()

		# convert data to zip if not done already
		if not pathlib.Path(zip_data).exists():
			self._convert_data(
				train_csv, dir_imgs, zip_data, imgs_extension, dtype,
				print_progress, n_imgs
			)

		# load data
		self.imgs, self.bounds, self.touch, self.masks, self.map = \
			self._load_data(zip_data, imgs_extension, n_imgs, dtype)

	# sanity check that every img, mask exist in map

	# self._sanity_check(data_ids=self.ids_map, img_ids=self.pths.keys())

	def __len__(self):
		return len(self.map)

	def __getitem__(self, idx):
		img = self.imgs[self.map[idx]].unsqueeze(0)
		bounds = self.bounds[self.map[idx]].unsqueeze(0)
		touch = self.touch[self.map[idx]].unsqueeze(0)
		mask = self.masks[self.map[idx]].unsqueeze(0)
		if self.transforms is not None:
			img = self.transforms(img)
		if self.target_transforms is not None:
			bounds = self.target_transforms(bounds)
			touch = self.target_transforms(touch)
			mask = self.target_transforms(mask)
		return img, bounds, touch, mask

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
			cls, train_csv, dir_imgs, zip_data, imgs_extension, dtype,
			print_progress, n_imgs
	):
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
				annotations = Annotations(data, pth.stem, dtype)

				# add img to zipfile
				zf.write(pth, f'imgs/{filename}')

				# Add the annotation boundaries that touch each other to zip
				with zf.open(f'bounds/{filename}', 'w') as file:
					annotations.bounds.save(file, 'png', optimize=True)

				# Add the annotation boundaries that touch each other to zip
				with zf.open(f'touch/{filename}', 'w') as file:
					annotations.touch.save(file, 'png', optimize=True)

				# convert annotations to image and add it to zipfile
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
			imgs = {
				idx: cls._load_zip_img(
					zf, f'imgs/{idx}.{imgs_extension}', dtype
				)
				for idx in map_
			}
			bounds = {
				idx: cls._load_zip_img(
					zf, f'bounds/{idx}.{imgs_extension}', dtype
				)
				for idx in map_
			}
			touch = {
				idx: cls._load_zip_img(
					zf, f'touch/{idx}.{imgs_extension}', dtype
				)
				for idx in map_
			}
			masks = {
				idx: cls._load_zip_img(
					zf, f'masks/{idx}.{imgs_extension}', dtype
				)
				for idx in map_
			}

			return imgs, bounds, touch, masks, map_

	@staticmethod
	def _gray_img_to_col(gray_img: torch.tensor, color):
		return torch.cat([gray_img * c for c in color], dim=0)

	@staticmethod
	def _load_zip_img(zf, filename, dtype):
		with zf.open(filename, 'r') as file:
			return torch.as_tensor(np.array(
				PIL.Image.open(file)), dtype=dtype
			) / 255  # change dtype to inherit dtype of class

	@staticmethod
	def _sanity_check(data_ids, img_ids):
		# todo: needs to be extended to include map
		for data_id in data_ids:
			if data_id not in img_ids:
				raise Exception(f'no img found for id: {data_id}')
