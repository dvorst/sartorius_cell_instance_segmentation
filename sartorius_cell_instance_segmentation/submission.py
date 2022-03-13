import torch
import numpy as np
import dataclasses
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


@dataclasses.dataclass
class Slice:
	start: int = -1
	end: int = -1
	group_idx: int = -1
	row_idx: int = -1  # idx of the row in the mask


def batch_prediction_to_annotations(names: Tuple[str], masks: torch.Tensor, file_pth: Optional[str] = 'submission.csv'):
	""" exports the provided mask to the challenges desired format. """
	if masks.shape[1] != 1:
		raise ValueError('mask should only have 1 channel, not {}'.format(masks.shape[1]))
	masks = masks.squeeze(dim=1)  # remove channel
	masks = masks.numpy()
	masks = masks.astype(np.uint8)
	annotations_list = _batch_prediction_to_annotations_using_numbba(masks)
	if file_pth is not None:
		with open(file_pth, 'w') as file:
			for n, annotations in zip(names, annotations_list):
				for a in annotations:
					file.write('{},{}\n'.format(n, a))
	return annotations_list


def _batch_prediction_to_annotations_using_numbba(masks):
	annotations_per_mask = []
	for mask in masks:
		sliced_mask = _slice_mask(mask)
		grouped_slices = _group_slices(sliced_mask)
		annotations = _annotations_from_grouped_slices(grouped_slices, mask_width_px=mask.shape[-1])
		annotations_per_mask.append(annotations)
	return annotations_per_mask


def _slice_mask(mask):
	""" mask is a list of sliced-rows, each sliced-row is a list of slices """
	return [_slice_row(row, row_idx) for row_idx, row in enumerate(mask)]


def _slice_row(row, row_idx):
	prev_px = 0
	slices = []
	for idx_px, px in enumerate(row):  # loop through each pixel of row
		if px != prev_px:  # if there is a change in pixel value, the slice starts or ends
			if px == 1:  # start of slice
				s = Slice(start=idx_px, row_idx=row_idx)  # create and append slice
			else:  # end of slice
				s.end = idx_px
				slices.append(s)
			prev_px = px
	if prev_px == 1:  # it is possible that the last pixel was a 1, in which case the end-pixel was not set
		s.end = len(row)  # set end of slice to number of pixels in the row
		slices.append(s)
	return slices


def _has_group(s: Slice) -> bool:
	return not _has_no_group(s)


def _has_no_group(s: Slice) -> bool:
	return s.group_idx == -1


def _slice1_is_left_of_slice2(slice1: Slice, slice2: Slice) -> bool:
	return slice1.end < slice2.start


def _slice1_is_right_of_slice2(slice1: Slice, slice2: Slice) -> bool:
	return slice1.start > slice2.end


def _touch(slice1: Slice, slice2: Slice) -> bool:
	# slices do not touch if they are situated neither to the left nor to the right of each other
	return not (
			_slice1_is_left_of_slice2(slice1, slice2) or
			_slice1_is_right_of_slice2(slice1, slice2)
	)


def _add_to_new_group(groups: List[List[Slice]], s: Slice):
	if len(groups[-1]) == 0:  # add slice to the last group if the last group is empty
		s.group_idx = len(groups) - 1
		groups[-1].append(s)
	else:  # otherwise, append it as a new group
		s.group_idx = len(groups)
		groups.append([s])


def _add_to_existing_group(groups: List[List[Slice]], slice_without_group: Slice, slice_with_group: Slice):
	""" add slice_without_group to group of slice_with_group """
	group_idx = slice_with_group.group_idx
	slice_without_group.group_idx = group_idx
	groups[group_idx].append(slice_without_group)


def _merge_groups(groups: List[List[Slice]], slice1: Slice, slice2: Slice):
	i1, i2 = slice1.group_idx, slice2.group_idx
	if i1 == i2:
		return  # nothing to be merged
	groups[i1].extend(groups[i2])  # merge groups
	for s in groups[i2]:
		s.group_idx = i1  # update group indexes
	groups[i2] = []  # remove slices from old group


def assign_slices_to_group(groups: List[List[Slice]], slice1: Slice, slice2: Slice):
	if _has_no_group(slice1) and _has_no_group(slice2):
		_add_to_new_group(groups, slice1)
		_add_to_existing_group(groups, slice_without_group=slice2, slice_with_group=slice1)
	elif _has_group(slice1) and _has_no_group(slice2):
		_add_to_existing_group(groups, slice_without_group=slice2, slice_with_group=slice1)
	elif _has_no_group(slice1) and _has_group(slice2):
		_add_to_existing_group(groups, slice_without_group=slice1, slice_with_group=slice2)
	else:  # if both have a group
		_merge_groups(groups, slice1, slice2)


def _group_slices(sliced_mask: List[List[Slice]]):
	groups: List[List[Slice]] = [[]]
	# to group the slices, each sliced_row is compared to the next sliced_row
	for sliced_row, next_sliced_row in zip(sliced_mask[:-1], sliced_mask[1:]):
		for s in sliced_row:  # loop through all the slices present in row
			for slice_nxt_row in next_sliced_row:  # also loop through all the slices of the next row
				if _touch(s, slice_nxt_row):
					assign_slices_to_group(groups, s, slice_nxt_row)
				elif _slice1_is_right_of_slice2(slice1=slice_nxt_row, slice2=s):
					break  # no point in iterating through the other slices of next_sliced row
			if _has_no_group(s):  # in case slice is not adjacent to any other slices (stand-alone group)
				_add_to_new_group(groups, s)
	return groups


def _annotations_from_grouped_slices(grouped_slices: List[List[Slice]], mask_width_px: int):
	annotations = []
	for group in grouped_slices:
		annotation = _annotation_from_group(group, mask_width_px)
		if annotation is not None:
			annotations.append(annotation)
	return annotations


def _annotation_from_group(group: List[Slice], mask_width_px: int):
	annotation = np.array([(s.row_idx * mask_width_px + s.start, s.end - s.start) for s in group])
	if len(annotation) == 0:
		return None
	annotation = annotation[annotation[:, 0].argsort()]
	annotation = ' '.join([str(elm) for elm in annotation.reshape(-1)])
	return annotation


def _test():
	import datasets
	import torch.utils.data

	# dataset
	ds = datasets.SupervisedDataset(
		train_csv='../data/sartorius-cell-instance-segmentation/train.csv',
		dir_imgs='../data/sartorius-cell-instance-segmentation/train',
		zip_data='../data/SupervisedDataset.zip',
		force_convert=False,
		print_progress=False,
		n_imgs=10
	)

	# data-loader
	dl = torch.utils.data.DataLoader(
		dataset=ds,
		batch_size=3,
		shuffle=True,
		generator=torch.Generator().manual_seed(515563983)
	)

	# test slice_mask
	_test_slice_mask([ds[idx][4] for idx in range(len(ds))])

	# test from_mask
	_test_from_mask(dl)


def _test_from_mask(dl):
	import datasets.annotations
	print('testing "from_mask()"...')
	for _, _, _, _, masks, names in dl:
		annotations_list = batch_prediction_to_annotations(names, masks, file_pth=None)
		errors = torch.empty(masks.shape[0])
		for idx, (annotations, mask) in enumerate(zip(annotations_list, masks)):
			reconstructed_mask = torch.zeros(mask.shape)
			for a in annotations:
				reconstructed_mask += datasets.annotations.decompress_single(a, mask.shape[-1], mask.shape[-2])

			diff = torch.abs(reconstructed_mask - mask)
			if torch.sum(diff) == 0:
				continue
			__show_overlay(
				mask, reconstructed_mask, show=True,
				title='mask: blue, reconstructed_mask: red, overlap: pink'
			)
			errors[idx] = torch.sum(reconstructed_mask - mask)
			print('number of errors found when compressing/decompressing mask: ', int(errors[idx]))
	print('...pass')


def _test_slice_mask(masks: list):
	print('testing "_slice_mask()"...')
	for mask in masks:
		# check shape
		if len(mask.shape) != 3:
			raise ValueError(
				'masks must be sliced_rows of torch tensors that have 3 dimensions (CxWxH), not {} dimensions'.format(
					len(mask.shape)))
		# to sliced_rows
		mask = mask.squeeze(dim=0)  # remove channel
		mask = mask.numpy()
		mask = mask.astype(np.uint8)
		slices = _slice_mask(mask)
		mask = torch.from_numpy(mask)

		# reconstruct mask
		reconstruct_mask = torch.zeros(mask.shape, dtype=torch.int32)
		for idx_row, row_slices in enumerate(slices):
			for s in row_slices:
				reconstruct_mask[idx_row, s.start:s.end] = 1

		# calculate difference
		diff = np.abs(reconstruct_mask - mask)

		# continue if there is no difference
		if torch.sum(diff) == 0:
			continue

		print(torch.sum(diff))
		# if there is a difference, plot it. Original mask is displayed in blue and the difference in red
		__show_overlay(mask, reconstruct_mask, show=True)

	print('...no differences found')


def __show_overlay(img1, img2, show=False, title=None):
	""" displays overlay, img1 in blue and img2 in red, note that overlapping parts are pink """
	if isinstance(img1, torch.Tensor):
		img1 = img1.numpy()
	if isinstance(img1, torch.Tensor):
		img1 = img1.numpy()
	if len(img1.shape) > 2:
		img1 = np.squeeze(img1, axis=0)
	if len(img2.shape) > 2:
		img2 = np.squeeze(img2, axis=0)
	if len(img1.shape) != 2:
		raise ValueError('img1.shape must be [1xHxW] or [HxW], not {}'.format(img1.shape))
	if len(img2.shape) != 2:
		raise ValueError('img2.shape must be [1xHxW] or [HxW], not {}'.format(img2.shape))
	img1, img2 = np.abs(img1), np.abs(img2)
	img1 = np.stack([img1 * color_channel for color_channel in [0, 0, 1.]], axis=0)
	img2 = np.stack([img2 * color_channel for color_channel in [1., 0, 0]], axis=0)
	img = img1 + img2
	img = np.moveaxis(img, 0, -1)  # color axis should be at end, not at the start
	plt.figure()
	plt.imshow(img)
	if title is not None:
		plt.title(title)
	if show:
		plt.show()


if __name__ == "__main__":
	_test()
