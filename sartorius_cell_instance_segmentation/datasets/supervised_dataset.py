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
				print_progress
			)

		# load data
		self.imgs, self.bounds, self.masks, self.map = \
			self._load_data(zip_data, imgs_extension)

	# sanity check that every img, mask exist in map

	# self._sanity_check(data_ids=self.ids_map, img_ids=self.pths.keys())

	def __len__(self):
		return len(self.map)

	def __getitem__(self, idx):
		img = self.imgs[self.map[idx]]
		img = self.transforms(img) if self.transforms is not None else img
		mask = self.masks[self.map[idx]]
		mask = self.target_transforms(
			mask) if self.transforms is not None else mask
		return img, mask

	@classmethod
	def overlay(
			cls, img, mask, col_img=(1., 1., 1.), col_mask=(1., 0., 0.),
			alpha=0.8
	):
		""" overlay mask on image """
		# convert img/mask to colored images
		img = cls._gray_img_to_col(img, col_img)
		mask = cls._gray_img_to_col(mask, col_mask)
		# return as PIL image
		return cls.tensor_to_pil(img * alpha + mask * (1 - alpha))

	@staticmethod
	def tensor_to_pil(t: torch.tensor):
		""" transform tensor with values between 0 and 1 to PIL image """
		return torchvision.transforms.functional.to_pil_image(
			(t * 255).type(torch.uint8)
		)

	@classmethod
	def _convert_data(
			cls, train_csv, dir_imgs, zip_data, imgs_extension, dtype,
			print_progress
	):
		""" Convert and save data provided by the challenge as a zipfile """

		print('SupervisedDataset: converting data...')

		# load train.csv file as pandas dataframe
		data = pd.read_csv(train_csv)

		# get paths of imgs located in dir_imgs
		pths = list(pathlib.Path(dir_imgs).glob('*.' + imgs_extension))

		# create data zip
		with zipfile.ZipFile(zip_data, 'w') as zf:
			# loop through pths
			for idx, pth in enumerate(pths):
				# the imgs, masks and bounds have the same filename
				filename = pth.stem + '.' + imgs_extension

				# convert data to annotations
				annotations = Annotations(data, pth.stem, dtype)

				# add img to zipfile
				zf.write(pth, 'imgs/' + filename)

				# Add the annotation boundaries that touch each other to zip
				with zf.open('bounds/' + filename, 'w') as file:
					annotations.bounds.save(file, 'png', optimize=True)

				# convert annotations to image and add it to zipfile
				with zf.open('masks/' + filename, 'w') as file:
					annotations.mask.save(file, 'png', optimize=True)

				# print progress
				print('SupervisedDataset: \tconverted ' + str(pth))

	@classmethod
	def _load_data(cls, zip_data, imgs_extension):
		""" load zip file data """

		# open zipfile
		with zipfile.ZipFile(zip_data, 'r') as zf:
			# extract filenames to create map = idx_str[idx_int]
			map_ = list(np.sort(np.unique(
				[pathlib.Path(file).stem for file in zf.namelist()]
			)))

			# read imgs, bounds and masks as dict
			imgs = {
				idx: cls._load_zip_img(
					zf, 'imgs/' + idx + '.' + imgs_extension
				)
				for idx in map_
			}
			bounds = {
				idx: cls._load_zip_img(
					zf, 'bounds/' + idx + '.' + imgs_extension
				)
				for idx in map_
			}
			masks = {
				idx: cls._load_zip_img(
					zf, 'masks/' + idx + '.' + imgs_extension
				)
				for idx in map_
			}

			return imgs, bounds, masks, map_

	@staticmethod
	def _gray_img_to_col(gray_img: torch.tensor, color):
		return torch.stack([gray_img * c for c in color])

	@staticmethod
	def _load_zip_img(zf, filename):
		with zf.open(filename, 'r') as file:
			return torch.as_tensor(np.array(
				PIL.Image.open(file), dtype=float)
			) / 255  # change dtype to inherit dtype of class

	@staticmethod
	def _sanity_check(data_ids, img_ids):
		# todo: needs to be extended to include map
		for data_id in data_ids:
			if data_id not in img_ids:
				raise Exception('no img found for id: ' + data_id)
