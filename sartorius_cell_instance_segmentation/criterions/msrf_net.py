"""
original code: https://github.com/amlarraz/MSRF-Net_PyTorch/blob/master/msrf.py
"""
import torch
from torch import nn
import torch.nn.functional as func


class CombinedLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.ce_loss = nn.CrossEntropyLoss()
		self.dice_loss = DiceLoss()
		self.bce_loss = nn.BCEWithLogitsLoss()

	def forward(self, input_, target):
		pred, edge_out, ds0, ds1 = input_
		msk, canny_label = target
		msk_sq = msk.squeeze().type(torch.int64)
		loss_pred_1 = self.ce_loss(ds0, msk) + self.dice_loss(ds0, msk_sq)
		loss_pred_2 = self.ce_loss(ds1, msk) + self.dice_loss(ds1, msk_sq)
		loss_pred_3 = self.ce_loss(pred, msk) + self.dice_loss(pred, msk_sq)
		loss_edge = self.bce_loss(edge_out, canny_label)
		loss = loss_pred_3 + loss_pred_1 + loss_pred_2 + loss_edge
		return loss


class DiceLoss(nn.Module):
	@staticmethod
	def forward(inp, target):
		""" dice coefficient, given there are only 2 classes. Dice of background is not calculated, only foreground. """
		cardinality = torch.sum(inp) + torch.sum(target)  # todo: shouldn't sum be mean?
		if cardinality == 0:  # prevent division by zero
			raise ValueError('division by zero')
		intersection = torch.sum(inp * target)  # todo: shouldn't sum be mean?
		return 1. - 2. * intersection / cardinality
