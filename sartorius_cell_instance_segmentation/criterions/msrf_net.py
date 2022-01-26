"""
original code: https://github.com/amlarraz/MSRF-Net_PyTorch/blob/master/msrf.py
"""
import torch
from torch import nn
import torch.nn.functional as func


def one_hot(labels, num_classes, device, dtype, eps=1e-6):
	r"""Converts an integer label 2D tensor to a one-hot 3D tensor.
	Args:
		labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
			where N is batch siz. Each value is an integer representing
			correct classification.
		num_classes (int): number of classes in labels.
		device (Optional[torch.device]): the desired device of returned tensor.
			if None, uses the current device for the default tensor type
			(see torch.set_default_tensor_type()). device will be the CPU
			for CPU tensor types and the current CUDA device for CUDA
			tensor types.
		dtype (Optional[torch.dtype]): the desired data type of returned
			tensor. Infers data type from values by default.
		eps:
	Returns:
		torch.Tensor: the labels in one hot tensor.
	Examples::
		>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
		>> tgm.losses.one_hot(labels, num_classes=3)
		>> tensor([[[ [1., 0.],
				[0., 1.]],
				[[0., 1.],
				[0., 0.]],
				[[0., 0.],
				[1., 0.]]]])
	"""
	if not torch.is_tensor(labels):
		raise TypeError(
			f"Input labels type is not a torch.Tensor. Got {type(labels)}"
		)
	if not len(labels.shape) == 3:
		raise ValueError(
			f"Invalid depth shape, we expect BxHxW. Got: {labels.shape}"
		)
	if not labels.dtype == torch.int64:
		raise ValueError(
			"labels must be of the same dtype torch.int64. Got: {}".format(
				labels.dtype))
	if num_classes < 1:
		raise ValueError(
			"The number of classes must be bigger than one."
			" Got: {}".format(num_classes)
		)
	batch_size, height, width = labels.shape
	one_hot_ = torch.zeros(
		batch_size, num_classes, height, width,
		device=device, dtype=dtype
	)
	return one_hot_.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class DiceLoss(nn.Module):
	r"""Criterion that computes Sørensen-Dice Coefficient loss.
	According to [1], we compute the Sørensen-Dice Coefficient as follows:
	.. math::
		\text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
	where:
		- :math:`X` expects to be the scores of each class.
		- :math:`Y` expects to be the one-hot tensor with the class labels.
	the loss, is finally computed as:
	.. math::
		\text{loss}(x, class) = 1 - \text{Dice}(x, class)
	[1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
	Shape:
		- Input: :math:`(N, C, H, W)` where C = number of classes.
		- Target: 	:math:`(N, H, W)` where each value is
					:math:`0 ≤ targets[i] ≤ C−1`.
	Examples:
		N = 5  # num_classes
		loss = tgm.losses.DiceLoss()
		inp = torch.randn(1, N, 3, 5, requires_grad=True)
		target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
		output = loss(input, target)
		output.backward()
	"""

	def __init__(self) -> None:
		super(DiceLoss, self).__init__()
		self.eps: float = 1e-6

	def forward(
			self,
			inp: torch.Tensor,
			target: torch.Tensor
	) -> torch.Tensor:
		if not torch.is_tensor(inp):
			raise TypeError(
				f"Input type is not a torch.Tensor. Got {type(inp)}"
			)
		if not len(inp.shape) == 4:
			raise ValueError(
				f"Invalid input shape, we expect BxNxHxW. Got: {inp.shape}"
			)
		if not inp.shape[-2:] == target.shape[-2:]:
			raise ValueError(
				"input and target shapes must be the same. Got: {}"
				"".format(inp.shape, inp.shape)
			)
		if not inp.device == target.device:
			raise ValueError(
				"input and target must be in the same device. Got: {}"
				"".format(inp.device, target.device)
			)
		# compute softmax over the classes axis
		input_soft = func.softmax(inp, dim=1)[:, 1:]

		# create the labels one hot tensor
		target_one_hot = one_hot(
			target, num_classes=inp.shape[1],
			device=inp.device, dtype=inp.dtype
		)[:, 1:]

		# compute the actual dice score
		dims = (1, 2, 3)
		intersection = torch.sum(input_soft * target_one_hot, dims)
		cardinality = torch.sum(input_soft + target_one_hot, dims)

		dice_score = 2. * intersection / (cardinality + self.eps)
		return torch.mean(1. - dice_score)


class CombinedLoss(nn.Module):
	def __init__(self, class_weights=None):
		super().__init__()
		self.ce_loss = nn.CrossEntropyLoss(class_weights)
		self.dice_loss = DiceLoss()
		self.bce_loss = nn.BCEWithLogitsLoss()

	def forward(self, input_, target):
		pred_3, pred_canny, pred_1, pred_2 = input_
		msk, canny_label = target
		msk = msk.squeeze().type(torch.int64)
		loss_pred_1 = self.ce_loss(pred_1, msk) + self.dice_loss(pred_1, msk)
		loss_pred_2 = self.ce_loss(pred_2, msk) + self.dice_loss(pred_2, msk)
		loss_pred_3 = self.ce_loss(pred_3, msk) + self.dice_loss(pred_3, msk)
		loss_canny = self.bce_loss(pred_canny, canny_label)
		loss = loss_pred_3 + loss_pred_1 + loss_pred_2 + loss_canny
		return loss
