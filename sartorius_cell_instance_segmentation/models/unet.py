"""
Stage   Resolution  scale   Padding (all/both sides)
0       520 x 704   4       0
1       130 x 176   4       (3,0)
2        34 x  44   2       (1,0)
3        18 x  22   2       0
4         9 x  11   -       -
"""
import torch.nn as nn
import torch


def print_resolutions(dataset, chs, pad):
	import numpy as np
	h, w = dataset[0][0].shape[1:]
	d = (1, 1)  # dilation
	k = (1, 1)  # kernel_size
	s = (1, 1)  # stride
	po = (0, 0)  # output padding
	t = False  # transpose
	pad = (*pad, 0)  # add 0 to end, otherwise the last layer (with no pooling) is missing
	params_prev = h * w * chs[1]
	print('i    w      h      ci     co     params   reduce')
	for idx, (ci, co, p) in enumerate(zip(chs[:-1], chs[1:], pad)):
		p = (p, p) if not isinstance(p, tuple) else p
		params = h * w * co
		print(
			'%i %6.1f %6.1f %6.1f %6.1f % 10.1f % 3.1f' %
			(idx, w, h, ci, co, params, params / params_prev)
		)
		params_prev = params
		if not t:
			h = np.floor((h + 2 * p[0] - d[0] * (k[0] - 1) - 1) / s[0] + 1)
			w = np.floor((w + 2 * p[1] - d[1] * (k[1] - 1) - 1) / s[1] + 1)
		else:
			h = (h - 1) * s[0] - 2 * p[0] + d[0] * (k[0] - 1) + po[0] + 1
			w = (w - 1) * s[1] - 2 * p[1] + d[1] * (k[1] - 1) + po[1] + 1
		h /= 4
		w /= 4


class UNet(nn.Module):
	def __init__(self, dataset=None, prnt: bool = False):
		""" dataset only has to be provided if print is true, which prints the size of each layer """
		super().__init__()
		chs = (1, 16, 64, 128, 256, 512)
		pad = (0, (3, 0), (1, 0), 0)
		scales = (4, 4, 2, 2)
		if prnt:
			print_resolutions(dataset, chs, pad)
		self.enc = Encoder(chs, pad, scales)
		self.dec = Decoder(chs, pad, scales)

	def forward(self, x, _):
		x = self.enc(x)
		x = self.dec(x)
		return x


class Encoder(nn.Module):
	def __init__(self, chs, pad, scales):
		super().__init__()
		self.blocks = nn.ModuleList([Block(ci, co) for ci, co in zip(chs[:-1], chs[1:])])
		self.pools = nn.ModuleList([
			nn.Sequential(
				ZeroPad(p),
				nn.MaxPool2d(s)
			) for s, p in zip(scales, pad)
		])

	def forward(self, x):
		x = self.blocks[0](x)
		out = [x]
		for idx, pool in enumerate(self.pools):
			x = pool(x)
			x = self.blocks[idx + 1](x)
			out.append(x)
		return out


class ZeroPad(nn.ConstantPad2d):
	def __init__(self, p):
		p = (p, p) if not isinstance(p, tuple) else p
		super().__init__((p[1], p[1], p[0], p[0]), 0.)


class Decoder(nn.Module):
	def __init__(self, chs, pad, scales):
		super().__init__()
		chs, pad, scales = chs[::-1], pad[::-1], scales[::-1]  # swap chs & pad from left to right
		self.blocks = nn.ModuleList([])
		for idx, (ci, co) in enumerate(zip(chs[:-1], chs[1:])):
			if idx == 0:  # bottleneck block
				self.blocks.append(Block(ci, co))
			elif idx == len(chs) - 1:  # last block
				self.blocks.append(nn.Sequential(
					self.conv(2 * ci, ci, (3, 3), padding=1, bias=True),
					self.conv(ci, co, (1, 1), padding=1, bias=True)
				))
			else:  # all other blocks
				self.blocks.append(Block(2 * ci, co))
		self.ups = nn.ModuleList([
			nn.ConvTranspose2d(c, c, kernel_size=(s, s), stride=(s, s), padding=p)
			for c, p, s in zip(chs[1:-1], pad, scales)
		])

	def forward(self, enc_out):
		enc_out = enc_out[::-1]  # swap tuple left to right
		x = self.blocks[0](enc_out[0])
		for idx, up in enumerate(self.ups):
			x = up(x)
			x = torch.cat([enc_out[idx + 1], x], dim=1)
			x = self.blocks[idx + 1](x)
		return x


class Block(nn.Module):
	def __init__(self, ci, co):
		super().__init__()
		self.cbr1 = CBR(ci, co)
		self.cbr2 = CBR(co, co)

	def forward(self, x):
		x = self.cbr1(x)
		x = self.cbr2(x)
		return x


class CBR(nn.Module):
	""" Conv2d + Batch-normalization + Relu """

	def __init__(self, ci, co):
		super().__init__()
		self.conv = nn.Conv2d(ci, co, (3, 3), padding=1, bias=False)
		self.bn = nn.BatchNorm2d(co)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x
