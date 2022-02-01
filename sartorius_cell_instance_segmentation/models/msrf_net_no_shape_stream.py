"""
original code: https://github.com/NoviceMAn-prog/MSRF-Net

Note that there are numerous alterations compared to the original code.
"""

from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as func


class MSRF(nn.Module):
	""" https://arxiv.org/pdf/2105.07451v1.pdf """

	def __init__(self, ci, n_classes, feat=32):
		""" ci: in_channels, feat: features"""
		super().__init__()

		self.encoder = Encoder()  # todo
		self.msrf_subnet = MSRFSubNet()  # todo
		self.decoder = Decoder()  # todo
		self.shape_stream = ShapeStream()  # todo
		self.concatenation = Concatenation()  # todo
		self.c3x3 = None  # todo
		self.c1x1 = None  # todo

	def forward(self, x, image_gradients):
		x = self.encoder(x)  # x = [E1, E2, E3, E4]
		x = self.msrf_subnet(x)  # x contains all four outputs fot he MSRF subnet block
		x_decoder, ds0, ds1 = self.decoder(x)  # see docstring of function
		x, edge_out = self.shape_stream(x, image_gradients)
		x = self.concatenation(x)
		x = self.c3x3(x)
		x = self.c1x1(x)

		return x, ds0, ds1, edge_out


class Encoder(nn.Module):
	def __init__(self, ci):
		super().__init__()
		c = [32, 64, 128, 256]
		self.e1 = EncoderBlock(ci=ci, co=c[0], pool=True, residual=False)
		self.e2 = EncoderBlock(ci=c[0], co=c[1], pool=True, residual=True)
		self.e3 = EncoderBlock(ci=c[1], co=c[2], pool=True, residual=True)
		self.e4 = EncoderBlock(ci=c[2], co=c[3], pool=False, residual=False)

	def forward(self, x):
		x, e1 = self.e1(x)
		x, e2 = self.e2(x)
		x, e3 = self.e3(x)
		x = self.e4(x)
		return e1, e2, e3, x


class EncoderBlock(nn.Module):
	def __init__(self, ci, co, pool: bool, residual: bool):
		""" ci: in_channels, co: out_channels, pool: whether to use maxpool, residual: whether to use res block """
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(ci, co, (3, 3), (1, 1), bias=True),
			nn.ReLU(),
			nn.Conv2d(co, co, (3, 3), (1, 1), bias=True),
			nn.ReLU(),
			nn.BatchNorm2d(co),
			SqueezeExcite(co),
		)
		if residual:
			self.residual = nn.Sequential(
				nn.Conv2d(co, co, (3, 3), (1, 1), bias=False),
				nn.ReLU(),
				nn.BatchNorm2d(co),
			)
		else:
			self.residual = None
		if pool:
			self.pool = nn.Sequential(
				nn.MaxPool2d((2, 2)),
				nn.Dropout2d(p=0.2),
			)
		else:
			self.pool = None

	def forward(self, x):
		x = self.block1(x)
		if self.residual is not None:
			x += self.residual(x)
		if self.pool is not None:
			x = self.pool(x)
		return x


class SqueezeExcite(nn.Module):
	""" https://arxiv.org/pdf/1709.01507v4.pdf """

	def __int__(self, c, ratio=16):
		super().__init__()
		self.squeeze = nn.Sequential(
			GlobalAvgPool2d(),
			nn.Linear(c, c // ratio),
			nn.ReLU(),
			nn.Linear(c // ratio, c),
			nn.Sigmoid(),
		)

	def forward(self, x):
		return x * self.squeeze(x)


class GlobalAvgPool2d(nn.Module):
	def __int__(self):
		super().__init__()

	@staticmethod  # todo: check if this interferes with backprop
	def forward(x):
		return x.mean(dim=[2, 3]).unsqueeze(dim=[2])  # todo: check this


class MSRFSubNet(nn.Module):
	""" https://arxiv.org/pdf/2105.07451v1.pdf """

	def __init__(self, n_blocks: int = 7):
		super().__init__()
		if n_blocks not in [7, 10]:
			raise Exception(f'Current implementation only supports 7 or 9 blocks, not {n_blocks}')
		self.dsdf_blocks = nn.ModuleList([])
		for idx in range(n_blocks):
			self.dsdf_blocks.append(DSDFBlock())  # todo

	def forward(self, x):
		y0, y1 = self.dsdf_blocks[0](x[0], x[1])
		y2, y3 = self.dsdf_blocks[1](x[2], x[3])
		y0, y1 = self.dsdf_blocks[2](y0, y1)
		y2, y3 = self.dsdf_blocks[3](y2, y3)
		y1, y2 = self.dsdf_blocks[4](y1, y2)
		y0, y1 = self.dsdf_blocks[5](y0, y1)
		y2, y3 = self.dsdf_blocks[6](y2, y3)
		if len(self.dsdf_blocks) == 9:
			y1, y2 = self.dsdf_blocks[7](y1, y2)
			y0, y1 = self.dsdf_blocks[8](y0, y1)
			y2, y3 = self.dsdf_blocks[9](y2, y3)
		# y1, y2, y3, y4 = y1 * 0.4, y2 * 0.4, y3 * 0.4, y4 * 0.4   # implemented in original, is removed
		x[0] += y0
		x[1] += y1
		x[2] += y2
		x[3] += y3
		return x


class ShapeStream(nn.Module):
	""" https://arxiv.org/pdf/1907.05740v1.pdf """

	def __init__(self, size, ci=None, ch=None):
		""" ci: in_channels, ch: channels hidden """
		super().__init__()  # todo
		if ch is None:
			ch = [64, 32, 16, 8]
		if ci is None:
			ci = [32, 64, 128, 256]

		self.conv = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(c, 1, (1, 1)),  # todo
				Interpolate(size=size)
			) for c in ci
		])

		self.res = nn.ModuleList([
			nn.Sequential(
				ResBlock(ci),  # todo
				Interpolate(size=size),
				nn.Conv2d(ci, co, (1, 1))
			) for ci, co in zip(ch[:-1], ch[1:])
		])

		self.gc = nn.ModuleList([
			nn.Sequential(
				Concatenation(),
				GatedConvolution(c)
			) for c in ch[1:]
		])

		self.out1 = nn.Sequential(
			nn.Conv2d(ch[-1], 1, (1, 1), bias=False),  # todo
			Interpolate(size),  # todo
			nn.Sigmoid()
		)
		self.out2 = nn.Sequential(
			Concatenation(),
			nn.Conv2d(2, 1, (1, 1), bias=False),
			nn.Sigmoid()
		)

	def forward(self, x: list, image_gradients):
		edge_out = self.conv[0](x[0])
		for x, conv, res, gc in zip(x[1:], self.conv[1:], self.res, self.gc):
			edge_out = res(edge_out)
			edge_out = gc(edge_out, conv(x))
		edge_out = self.out1(edge_out)
		x = self.out2(edge_out, image_gradients)
		return x, edge_out  # edge_out is used to calculate edge_loss


class Interpolate(nn.Module):
	""" This class might as well be replaced by interpolate, but by making a separate Module for this, I find it makes
	the code more readable"""

	def __init__(self, size):
		super().__init__()
		self.size = size

	def forward(self, x):
		return nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=True)


class ResBlock(nn.Module):
	""" https://arxiv.org/pdf/1512.03385v1.pdf """

	def __init__(self, c):
		""" c: channels """
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(c, c, (3, 3), padding=1, bias=False),
			nn.BatchNorm2d(c),
			nn.ReLU(),
			nn.Conv2d(c, c, (3, 3), padding=1, bias=False),
			nn.BatchNorm2d(c),
		)
		self.act = nn.ReLU()

	def forward(self, x):
		x += self.layers(x)
		x = self.act(x)
		return x


class GatedConvolution(nn.Module):
	""" https://arxiv.org/pdf/1612.08083v3.pdf """

	def __init__(self, c):
		""" c: channels """
		super().__init__()  # todo
		self.gate = nn.Sequential(
			Concatenation(),
			nn.BatchNorm2d(c + 1),
			nn.Conv2d(c + 1, c + 1, (1, 1)),  # todo: decide whether to add bias=False (despite no BN applied after)
			nn.ReLU(),
			nn.Conv2d(c, c + 1, (1, 1), bias=False),
			nn.BatchNorm2d(c + 1),
			nn.Sigmoid()
		)
		self.conv = nn.Conv2d(c + 1, c, (1, 1), bias=False)

	def forward(self, x1, x2):
		x2 = self.gate(x1, x2)
		x1 = x1 * x2
		del x2  # todo: is the effect of this negligible?
		x1 = self.conv(x1)
		return x1


class Concatenation(torch.nn.Module):
	""" This class might as well be replaced by torch.cat, but by making a separate Module for this I find it makes
	the code more readable"""

	def __init__(self):
		super().__init__()

	@staticmethod  # todo: does this interfere with backward propagation?
	def forward(x1, x2):
		return torch.cat((x1, x2), dim=1)


class Decoder(nn.Module):
	def __init__(self):
		super().__init__()  # todo
		self.d2 = DecoderBlock()  # todo
		self.d3 = DecoderBlock()  # todo
		self.d4 = DecoderBlock()  # todo

	def forward(self, x):
		ds0 = self.d2(x[2:3])
		ds1 = self.d3(x[1])
		x = self.d4[x[0]]
		return x, ds0, ds1


class DecoderBlock(nn.Module):
	def __init__(self):
		super().__init__()  # todo

	def forward(self, x):
		return None  # todo
