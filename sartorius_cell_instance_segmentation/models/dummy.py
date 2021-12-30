import torch.nn as nn
import torch


class DummyNet(nn.Module):

	def __init__(self, co):
		"""
		:param co: #channels output
		"""
		super().__init__()
		ci = 2
		self.dummy_layer = nn.Sequential(
			nn.Conv2d(ci, co, (3, 3), padding=1),
			nn.ReLU()
		)

	def forward(self, img, canny):
		x = torch.cat([img, canny], dim=1)
		x = self.dummy_layer(x)
		return x, x[:, 0, :, :].unsqueeze(1), x, x
