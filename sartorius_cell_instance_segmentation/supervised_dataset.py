import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional
from torchvision.models.resnet import BasicBlock
import PIL
import matplotlib.pyplot as plt
import pathlib
import zipfile


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
			force_convert=False
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
				train_csv, dir_imgs, zip_data, imgs_extension, dtype
			)

		# load data
		self.imgs, self.masks, self.map = \
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
	def _convert_data(
			cls, train_csv, dir_imgs, zip_data, imgs_extension, dtype
	):
		"""
		Convert data as provided by the challenge into something more
		convenient
		"""

		# load train.csv file as pandas dataframe
		data = pd.read_csv(train_csv)

		# get paths of imgs located in dir_imgs
		pths = pathlib.Path(dir_imgs).glob('*.' + imgs_extension)

		# create data zip
		with zipfile.ZipFile(zip_data, 'w') as zf:

			# loop through pths
			for idx, pth in enumerate(pths):

				# add img to zipfile
				zf.write(pth, 'imgs/' + pth.stem + '.' + imgs_extension)

				# extract mask of image from data.csv and save it to zipfile
				mask_pth = 'masks/' + pth.stem + '.' + imgs_extension
				with zf.open(mask_pth, 'w') as file:
					mask = cls._extract_mask(data, pth.stem, dtype)
					mask.save(file, 'png', optimize=True)

				if idx > 1:
					return

	@classmethod
	def _load_data(cls, zip_data, imgs_extension):
		""" load zip file data """

		# open zipfile
		with zipfile.ZipFile(zip_data, 'r') as zf:
			# get all imgs/masks present in zipfile
			files = zf.namelist()

			# extract filenames to create map = idx_str[idx_int]
			map_ = list(
				np.sort(
					np.unique([pathlib.Path(file).stem for file in files])))

			# read imgs and masks
			imgs = {
				idx: cls._load_zip_img(
					zf,
					'imgs/' + idx + '.' + imgs_extension
				) for idx in map_
			}
			masks = {
				idx: cls._load_zip_img(
					zf, 'masks/' + idx + '.' + imgs_extension
				) for idx in map_
			}

			return imgs, masks, map_

	def overlay(self, idx, col1=(1., 1., 1.), col2=(1., 0., 0.), alpha=0.9):
		# get img and mask that correspond to idx
		img, mask = self.__getitem__(idx)

		# convert these to colored images
		img = self._gray_img_to_col(img, col1),
		mask = self._gray_img_to_col(mask, col2)
		# return as PIL image
		return torchvision.transforms.functional.to_pil_image(
			img * alpha + mask * (1 - alpha))

	@staticmethod
	def _gray_img_to_col(gray_img, color):
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

	@staticmethod
	def _decompress_annotation(annotation_string, width, height, dtype):
		# convert annotation string to matrix:
		# [column 1: pixel index, column 2: length]
		annotation_tensor = torch.tensor(
			[int(elm) for elm in annotation_string.split(' ')]
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

	@classmethod
	def _get_annotations(cls, data, idx_img, dtype):
		# get all annotations from <data> that belong to <idx>
		return torch.stack(
			[
				cls._decompress_annotation(
					row['annotation'],
					row['width'],
					row['height'],
					dtype
				)
				for _, row in data.loc[data['id'] == idx_img].iterrows()
			]
		)

	@staticmethod
	def _correct_mask(mask, annotations):
		return mask  # todo: change

	@classmethod
	def _extract_mask(cls, data, idx_img: str, dtype):
		""" extract mask from data belonging to idx:str """
		# get all annotations belonging to img_idx
		annotations = cls._get_annotations(data, idx_img, dtype)

		# loop through annotations to convert it to mask
		mask = None
		for idx_annotation, annotation in enumerate(annotations):
			mask = mask + annotation if mask is not None else annotation

		# some annotations overlap, these need to be corrected
		mask = cls._correct_mask(mask, annotations)

		# todo: delete below after overlapping annotations are corrected for
		mask = mask / torch.max(mask)

		# return mask as PIL image
		return torchvision.transforms.functional.to_pil_image(mask)
