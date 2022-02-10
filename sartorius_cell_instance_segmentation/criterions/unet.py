import torch


class UNet(torch.nn.BCEWithLogitsLoss):
	def __init__(self):
		super().__init__()

	def forward(self, inp, target):
		return super().forward(inp, target[0])
