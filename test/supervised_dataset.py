import torch
import matplotlib.pyplot as plt
import sartorius_cell_instance_segmentation as scis


def supervised_dataset():
	# create ds
	ds = scis.datasets.SupervisedDataset(
		dtype=torch.float32,
		train_csv='data/sartorius-cell-instance-segmentation/train.csv',
		dir_imgs='data/sartorius-cell-instance-segmentation/train',
		zip_data='data/SupervisedDataset.zip',
		force_convert=True
	)

	# show first img and mask
	for img, mask in ds:
		# show img
		scis.util.imshow(img)

		# show mask
		scis.util.imshow(ds.tensor_to_pil(mask))

		# show overlay
		scis.util.imshow(ds.overlay(img, mask))

		break

	# loop through entire ds:
	for img, mask in ds:
		pass

	plt.show()
