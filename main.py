import test
import torch
import torchvision
import torchvision.transforms
import torch.utils.data
import matplotlib.pyplot as plt
import tensorboard
import tensorboard.program
import tensorboard.main
from numba import njit
import pstats
import cProfile
import pandas as pd
import numpy as np
from typing import Optional
import sartorius_cell_instance_segmentation as scis


# timing template
# with cProfile.Profile() as pr:
#     pass  # function here
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
# stats.print_stats()
# stats.dump_stats(filename='temp/t8.prof')


def main():
	"""
	Configuration
	"""
	split_pct = 0.8
	batch_size = 16
	dtype = torch.float32
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	epochs = 1

	"""
	Dataset
	"""
	torch.manual_seed(188990338)  # seed for random transformations
	transforms = None  # transforms = torchvision.transforms.Pad((0, 4))  # msrf-net
	target_transforms = transforms

	# complete dataset
	ds_train = scis.datasets.SupervisedDataset(
		transforms=transforms,
		target_transforms=target_transforms,
		dtype=dtype,
		train_csv='data/sartorius-cell-instance-segmentation/train.csv',
		dir_imgs='data/sartorius-cell-instance-segmentation/train',
		zip_data='data/SupervisedDataset.zip',
		force_convert=False,
		n_imgs=12
	)

	# train & valid dataset
	ds_train, ds_valid = torch.utils.data.random_split(
		dataset=ds_train,
		lengths=[int(split_pct * len(ds_train)), len(ds_train) - int(split_pct * len(ds_train))],
		generator=torch.Generator().manual_seed(711006933)
	)

	# test dataset
	ds_test = scis.datasets.TestDataset(dtype=dtype,dir_pth='data/sartorius-cell-instance-segmentation/test')

	# data-loaders
	dl_train = torch.utils.data.DataLoader(
		dataset=ds_train,
		batch_size=batch_size,
		shuffle=True,
		generator=torch.Generator().manual_seed(515563983)
	)
	dl_valid = torch.utils.data.DataLoader(
		dataset=ds_valid,
		batch_size=batch_size,
		shuffle=False
	)

	dl_test = torch.utils.data.DataLoader(
		dataset=ds_test,
		batch_size=batch_size,
		shuffle=False
	)

	"""
	Training
	"""
	model = scis.models.UNet(ds_train, prnt=False)
	# model = scis.models.MSRF(
	# 	c_enc=[16, 64, 128, 256], c_ss=[16, 16, 16, 8, 1], downscales_enc=[2, 2, 2, 2], n_msrf_block_layers=2,
	# 	downscale_ss=2,
	# )

	criterion = scis.criterions.UNet()
	# criterion = scis.criterions.MSRF()

	optimizer = torch.optim.Adam(model.parameters())

	summary_writer = scis.SummaryWriterExtended(n_imgs_per_epoch=10)

	# scis.train(
	# 	epochs=epochs,
	# 	dl_train=dl_train,
	# 	dl_valid=dl_valid,
	# 	model=model,
	# 	device=device,
	# 	optimizer=optimizer,
	# 	criterion=criterion,
	# 	summary_writer=summary_writer,
	# 	dtype=dtype
	# )

	"""
	Test
	"""
	scis.test(ds=dl_test, model=model, dst_pth_submission='submission.csv')


# def test_ds(ds, dl_train, dl_valid):
# 	print(f'{len(ds)=}')
# 	print(f'{len(dl_train)=}')
# 	print(f'{len(dl_valid)}')
# 	for img, canny, bounds, touch, mask in dl_train:
# 		print(f'{img.shape=}')
# 		print(f'{canny.shape=}')
# 		print(f'{bounds.shape=}')
# 		print(f'{touch.shape=}')
# 		print(f'{mask.shape=}')
#
# 		print(f'{img.dtype=}')
# 		print(f'{canny.dtype=}')
# 		print(f'{bounds.dtype=}')
# 		print(f'{touch.dtype=}')
# 		print(f'{mask.dtype=}')
#
# 		idx = 2
# 		scis.imshow(img[idx])
# 		scis.imshow(canny[idx])
# 		scis.imshow(bounds[idx])
# 		scis.imshow(touch[idx])
# 		scis.imshow(mask[idx])
# 		scis.imshow(ds.overlay(img[idx], bounds[idx], touch[idx], mask[idx]))
# 		plt.show()
# 		break




if __name__ == "__main__":
	main()
