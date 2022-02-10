"""
original code: https://github.com/amlarraz/MSRF-Net_PyTorch/blob/master/msrf.py
"""
import torch
from torch import nn
import torch.nn.functional as func


class MSRF(nn.Module):
	def __init__(self):
		super().__init__()
		self.dice_loss = DiceLoss()
		self.bce_loss = nn.BCEWithLogitsLoss()

	def forward(self, input_, target):
		pred, edge_out, ds0, ds1 = input_
		msk, canny_label = target
		loss_pred_1 = self.bce_loss(ds0, msk) + self.dice_loss(ds0, msk)
		loss_pred_2 = self.bce_loss(ds1, msk) + self.dice_loss(ds1, msk)
		loss_pred_3 = self.bce_loss(pred, msk) + self.dice_loss(pred, msk)
		loss_edge = self.bce_loss(edge_out, canny_label)
		loss = loss_pred_3 + loss_pred_1 + loss_pred_2 + loss_edge
		return loss


class DiceLoss(nn.Module):
	def __init__(self):
		super().__init__()
		self.sig = nn.Sigmoid()

	def forward(self, inp, target):
		""" dice coefficient, given there are only 2 classes. Dice of background is not calculated, only foreground. """
		inp = self.sig(inp)
		cardinality = torch.mean(inp) + torch.mean(target)
		if cardinality == 0:  # prevent division by zero
			raise ValueError('division by zero')
		intersection = torch.mean(inp * target)
		return 1. - 2. * intersection / cardinality
