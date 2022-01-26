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
	Dataset
	"""

	# configuration
	split_pct = 0.8
	batch_size = 2
	dtype = torch.float32
	#

	torch.manual_seed(188990338)  # seed for random transformations
	# transforms = torchvision.transforms.CenterCrop(size=128)
	transforms = None
	target_transforms = transforms

	# complete dataset
	ds = scis.datasets.SupervisedDataset(
		transforms=transforms,
		target_transforms=target_transforms,
		dtype=dtype,
		train_csv='data/sartorius-cell-instance-segmentation/train.csv',
		dir_imgs='data/sartorius-cell-instance-segmentation/train',
		zip_data='data/SupervisedDataset.zip',
		force_convert=False,
		n_imgs=50
	)

	# train & valid dataset
	ds_train, ds_valid = torch.utils.data.random_split(
		dataset=ds,
		lengths=[int(split_pct * len(ds)), len(ds) - int(split_pct * len(ds))],
		generator=torch.Generator().manual_seed(711006933)
	)

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

	# test_ds(ds=ds, dl_train=dl_train, dl_valid=dl_valid)

	"""
	Training
	"""

	# config
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	epochs = 3
	#

	# model = MSRF(in_ch=1, n_classes=2)
	model = scis.models.DummyUNet()

	# criterion = CombinedLoss()
	criterion = scis.criterions.BCELoss()

	optimizer = torch.optim.Adam(model.parameters())

	# test_model(model=model, dtype=dtype)

	scis.Train(
		epochs=epochs,
		dl_train=dl_train,
		dl_valid=dl_valid,
		model=model,
		device=device,
		optimizer=optimizer,
		criterion=criterion,
		dtype=dtype
	)


def test_ds(ds, dl_train, dl_valid):
	print(f'{len(ds)=}')
	print(f'{len(dl_train)=}')
	print(f'{len(dl_valid)}')
	for img, canny, bounds, touch, mask in dl_train:
		print(f'{img.shape=}')
		print(f'{canny.shape=}')
		print(f'{bounds.shape=}')
		print(f'{touch.shape=}')
		print(f'{mask.shape=}')

		print(f'{img.dtype=}')
		print(f'{canny.dtype=}')
		print(f'{bounds.dtype=}')
		print(f'{touch.dtype=}')
		print(f'{mask.dtype=}')

		idx = 2
		scis.imshow(img[idx])
		scis.imshow(canny[idx])
		scis.imshow(bounds[idx])
		scis.imshow(touch[idx])
		scis.imshow(mask[idx])
		scis.imshow(ds.overlay(img[idx], bounds[idx], touch[idx], mask[idx]))
		plt.show()
		break


def test_model(model, dtype):
	x = torch.randn((2, 1, 128, 128)).type(dtype)
	canny = torch.randn((2, 1, 128, 128)).type(dtype)
	out = model(x, canny)
	for o in out:
		print(o.shape)


if __name__ == "__main__":
	main()
