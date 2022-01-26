import torch


class BCELoss(torch.nn.BCELoss):
	def __init__(self):
		super().__init__()

	def forward(self, input_, target):
		return super().forward(input_[0], target[0])
