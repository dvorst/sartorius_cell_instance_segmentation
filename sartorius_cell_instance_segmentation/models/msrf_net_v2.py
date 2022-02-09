"""
original code: https://github.com/NoviceMAn-prog/MSRF-Net

Note that there are numerous alterations compared to the original code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import List
import sys


def cumulative_product(x):
	""" if x=[1,2,3,4] then this function returns [1, 1*2, 1*2*3, 1*2*3*4] """
	return torch.tensor([torch.prod(torch.tensor(x[:(idx + 1)])) for idx in range(len(x))])


class MSRF(nn.Module):
	""" See original paper [1] figure 2
	[1] https://arxiv.org/pdf/2105.07451v1.pdf"""

	def __init__(
			self,
			ci: int = 1,
			c_enc: List = None,
			c_ss: List = None,
			downscales_enc: List = None,
			downscale_ss: int = 1,
			n_msrf_blocks: int = 7,
			n_msrf_block_layers: int = 5
	):
		""" ci: in_channels, feat: features"""
		super().__init__()
		if c_enc is None:
			c_enc = [32, 64, 128, 256]
		if c_ss is None:
			c_ss = [32, 32, 16, 8, 1]
		if downscales_enc is None:
			downscales_enc = [1, 2, 2, 2]
		self.encoder = Encoder(ci, c_enc, scales=downscales_enc)
		self.msrf_subnet = MSRFSubNet(c_enc, n_blocks=n_msrf_blocks, n_block_layers=n_msrf_block_layers)
		self.decoder = Decoder(c_enc, scales=downscales_enc)
		self.shape_stream = ShapeStream(c_enc, ch=c_ss, scales_enc=downscales_enc, downscale_ss=downscale_ss)
		self.final = Final(scales=downscales_enc)

	def forward(self, x, image_gradients):
		x = self.encoder(x)  # x = [E1, E2, E3, E4]
		x = self.msrf_subnet(x)  # x contains all four outputs fot he MSRF subnet block
		x_ss, edge_out = self.shape_stream(x, image_gradients)
		x, ds0, ds1 = self.decoder(x)  # ds0: pred2 in original code, ds1: pred4 in original code
		x = self.final(x, x_ss)
		return x, edge_out, ds0, ds1


class Final(nn.Module):
	def __init__(self, scales):
		super().__init__()
		s = scales[0]
		self.conv_t = nn.ConvTranspose2d(1, 1, (s, s), stride=(s, s), bias=False) if s != 1 else None
		self.concatenation = Concatenation()
		self.c3x3 = nn.Conv2d(2, 1, (3, 3), padding=1)
		self.c1x1 = nn.Conv2d(1, 1, (1, 1))
		self.relu = nn.ReLU()

	def forward(self, x, x_ss):
		if self.conv_t is not None:
			x = self.conv_t(x)
		x = self.concatenation([x_ss, x])
		del x_ss
		x = self.c3x3(x)
		x = self.relu(x)
		x = self.c1x1(x)
		# x = self.relu(x) # nn.BCEWithLogitsLoss() is used to calculate loss, which uses sigmoid
		return x


class Encoder(nn.Module):
	def __init__(self, ci, co, scales, use_residuals=None):
		super().__init__()
		# c = [32, 64, 128, 256]
		if use_residuals is None:
			use_residuals = [True] * len(co)
			use_residuals.insert(0, False)
		ci = [ci, *co]
		self.encoder_blocks = nn.ModuleList([
			EncoderBlock(ci=ci[idx], co=c, residual=r, scale=s)
			for idx, (c, r, s) in enumerate(zip(co, use_residuals, scales))
		])

	def forward(self, x):
		e = []
		for encoder_block in self.encoder_blocks:
			x = encoder_block(x)
			e.append(x)
		return e


class EncoderBlock(nn.Module):
	def __init__(self, ci, co, residual: bool, scale: int = 2):
		""" ci: in_channels, co: out_channels, pool: whether to use maxpool, residual: whether to use res block """
		super().__init__()
		if scale < 1:
			# raise ValueError(f'only scales of 1 and larger are supported, not {scale=}')
			raise ValueError('only scales of 1 and larger are supported, not %i' % scale)
		self.residual = nn.Sequential(
			nn.Conv2d(ci, ci, (3, 3), padding=(1, 1), bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(ci),
		) if residual else None
		self.pool = nn.Sequential(
			nn.MaxPool2d((scale, scale)),
			nn.Dropout2d(p=0.2),
		) if scale != 1 else None
		self.layers = nn.Sequential(
			nn.Conv2d(ci, co, (3, 3), (1, 1), padding=(1, 1), bias=True),
			nn.ReLU(),
			nn.Conv2d(co, co, (3, 3), (1, 1), padding=(1, 1), bias=True),
			nn.ReLU(),
			nn.BatchNorm2d(co),
			SqueezeExcite(co),
		)

	def forward(self, x):
		if self.residual is not None:
			x = x + self.residual(x)
		if self.pool is not None:
			x = self.pool(x)
		x = self.layers(x)
		return x


class SqueezeExcite(nn.Module):
	""" https://arxiv.org/pdf/1709.01507v4.pdf """

	def __init__(self, c, ratio=16):
		super().__init__()
		self.squeeze = nn.Sequential(
			GlobalAvgPool2d(),
			Flatten(),
			nn.Linear(c, c // ratio),
			nn.ReLU(),
			nn.Linear(c // ratio, c),
			nn.Sigmoid(),
			Unflatten(),
		)

	def forward(self, x):
		return x * self.squeeze(x)


class GlobalAvgPool2d(nn.Module):
	@staticmethod  # todo: check if this interferes with backprop
	def forward(x):
		x = x.mean(dim=[2, 3])
		x = x.unsqueeze(dim=-1)
		x = x.unsqueeze(dim=-1)
		return x


class Flatten(nn.Module):
	@staticmethod  # todo: check if this interferes with backprop
	def forward(x):
		return torch.squeeze(x)


class Unflatten(nn.Module):
	@staticmethod
	def forward(x):
		return x.unsqueeze(dim=-1).unsqueeze(dim=-1)


class MSRFSubNet(nn.Module):
	"""
	See figure 1b of the original paper [1]. Note however, that the code published by the authors [2] contains a bug
	[3] resulting in the last three DSDF blocks displayed in figure not being used.

	[1] https://arxiv.org/pdf/2105.07451v1.pdf
	[2] https://github.com/NoviceMAn-prog/MSRF-Net
	[3] https://github.com/NoviceMAn-prog/MSRF-Net/issues/8
	"""

	def __init__(self, c: List, n_blocks: int = 7, n_block_layers: int = 5):
		""" c: output channels of [E1, E2, E3, E4] (encoder blocks), the output channels of SMRF are the same"""
		super().__init__()
		if n_blocks not in [7, 10]:
			# raise ValueError(f'Current implementation only supports 7 or 9 blocks, not {n_blocks}')
			raise ValueError('Current implementation only supports 7 or 9 blocks, not %i' % n_blocks)
		if len(c) != 4:
			# raise ValueError(f'Current implementation only supports 4 inputs, not {len(c)=}')
			raise ValueError('Current implementation only supports 4 inputs, not %i' % len(c))
		self.dsdf_blocks = nn.ModuleList([])
		for idx in range(n_blocks):
			if idx in [0, 2, 5, 8]:  # upper blocks
				self.dsdf_blocks.append(DSDFBlock(ch=c[0], cl=c[1], c=c[1] // 2, n_layers=n_block_layers))
			if idx in [1, 3, 6, 9]:  # lower blocks
				self.dsdf_blocks.append(DSDFBlock(ch=c[2], cl=c[3], c=c[1] // 2, n_layers=n_block_layers))
			elif idx in [4, 7]:  # center blocks
				self.dsdf_blocks.append(DSDFBlock(ch=c[1], cl=c[2], c=c[1] // 2, n_layers=n_block_layers))

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
		# y1, y2, y3, y4 = y1 * 0.4, y2 * 0.4, y3
		return x[0] + y0, x[1] + y1, x[2] + y2, x[3] + y3


class DSDFBlock(nn.Module):
	"""
	See figure 1a of the original paper [1]. Note however, that the code published by the authors [2] also uses xh and
	xl at the input of each Conv+LReLu layer [3]. This skip connection is not implemented, only the ones that are
	indicated in figure 1a.

	[1] https://arxiv.org/pdf/2105.07451v1.pdf
	[2] https://github.com/NoviceMAn-prog/MSRF-Net
	[3] https://github.com/NoviceMAn-prog/MSRF-Net/issues/19
	"""

	def __init__(self, ch, cl, c, n_layers: int = 5):
		""" ch: channels high input, cl: channels low input, c: hidden channels
		Note: high/low output channels will be the same as high/low inputs. """
		super().__init__()
		if n_layers < 1:
			# raise ValueError(f'n_layers must bbe larger than 1, not {n_layers=}')
			raise ValueError('n_layers must bbe larger than 1, not %i' % n_layers)
		self.layer_h, self.layer_l, self.down, self.up = \
			nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([]), nn.ModuleList([])
		for idx in range(n_layers):
			if idx == 0:  # first layer
				lh = DSDFLayer(ch, c, n_cat=idx + 1)
				ll = DSDFLayer(cl, c, n_cat=idx + 1)
			elif idx == n_layers - 1:  # last layer
				lh = DSDFLayer(c, ch, n_cat=idx + 1)
				ll = DSDFLayer(c, cl, n_cat=idx + 1)
			else:  # other layers
				lh = DSDFLayer(c, c, n_cat=idx + 1)
				ll = DSDFLayer(c, c, n_cat=idx + 1)
			self.layer_h.append(lh)
			self.layer_l.append(ll)

			if idx < n_layers - 1:  # the last high- and low-layer do not have to be up/down scaled
				self.down.append(DSDFScale(c, down=True))
				self.up.append(DSDFScale(c, up=True))

	def forward(self, xh, xl):
		cat_h, cat_l = [], []
		mh = self.layer_h[0](xh)
		ml = self.layer_l[0](xl)
		for idx, (up, down) in enumerate(zip(self.up, self.down)):
			cat_h.append(mh)
			cat_l.append(ml)
			# note, do not change line below to two separate lines, since ml uses mh as input
			mh, ml = self.layer_h[idx + 1]([*cat_h, up(ml)]), self.layer_l[idx + 1]([*cat_l, down(mh)])
		# mh *= 0.4  # this action is performed in original paper, but it shouldn't have any effect (not tested)
		# ml *= 0.4  # this action is performed in original paper, but it shouldn't have any effect (not tested)
		mh = mh + xh
		ml = ml + xl
		return mh, ml


class DSDFLayer(nn.Module):
	def __init__(self, ci, co, n_cat: int = 1):
		super().__init__()
		self.cat = Concatenation() if n_cat != 1 else None  # do not concatenate if there is only 1 input
		self.conv = nn.Conv2d(
			ci * n_cat, co, (3, 3), padding=(1, 1)
		)
		# todo: apply batch-norm and set bias=False in conv block?
		self.lrl = nn.LeakyReLU()

	def forward(self, x):
		if self.cat is not None:
			x = self.cat(x)
		x = self.conv(x)
		x = self.lrl(x)
		return x


class DSDFScale(nn.Module):
	def __init__(self, c, up: bool = False, down: bool = False):
		""" the code[2] of the original MSRF-net [1], scales using (transposed) convolutions with a 3x3 kernel size,
		which causes artifacts. Scaling is therefore implemented with a (2,2) kernel size instead.
		[1] https://arxiv.org/pdf/2105.07451v1.pdf
		[2] https://github.com/NoviceMAn-prog/MSRF-Net """
		super().__init__()
		if up:
			self.conv = nn.ConvTranspose2d(c, c, (2, 2), stride=(2, 2))
		elif down:
			self.conv = nn.Conv2d(c, c, (2, 2), stride=(2, 2))
		elif up and down:
			raise ValueError('up & bool cannot both be True, since this block either up-scales or down-scales')
		else:
			raise ValueError('either up or bool must be True, since this block will either up-scale or down-scale')
		# todo: apply batch-norm and set bias=False in conv block?
		self.lrl = nn.LeakyReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.lrl(x)
		return x


class ShapeStream(nn.Module):
	""" https://arxiv.org/pdf/1907.05740v1.pdf """

	def __init__(self, c, ch=None, scales_enc=None, downscale_ss: int = 1):
		""" ci: in_channels, ch: channels hidden """
		super().__init__()

		if len(c) != 4:
			# raise ValueError(f'Current implementation only supports 4 inputs, not {len(c)}=')
			raise ValueError('Current implementation only supports 4 inputs, not %i' % len(c))
		if len(ch) != len(c) + 1:
			# raise ValueError(f'number of hidden chan should be equal to input chan +1 {len(ch)=} != {len(c)+1=}')
			raise ValueError(
				'number of hidden chan should be equal to input chan +1, %i != %i' % (len(ch), len(c) + 1)
			)
		if len(scales_enc) != len(c):
			# raise ValueError(f'number of scales should be equal to input channels {len(scales)=} != {len(c)=}')
			raise ValueError('number of scales should be equal to input channels %i != %i' % (len(scales_enc), len(c)))

		scales_enc = cumulative_product(scales_enc) / downscale_ss

		self.conv_in = nn.ModuleList([])
		for idx, (c, scale) in enumerate(zip(c, scales_enc)):
			co = ch[0] if idx == 0 else 1  # every conv has only 1 output channel, except for the first one
			if scale == 1:  # skip up-sampling if scale is 1
				self.conv_in.append(nn.Conv2d(c, co, (1, 1)))
			# elif scale > 1:  # apply bilinear up/down-sampling if scale is larger than 1
			else:
				self.conv_in.append(nn.Sequential(
					nn.Conv2d(c, co, (1, 1)),
					nn.UpsamplingBilinear2d(scale_factor=scale)  # todo: replace by transpose conv?
				))
		# else:
		# 	raise ValueError('scale must be >=1, other scales are not supported')

		self.res = nn.ModuleList([
			nn.Sequential(
				ResBlock(ci),
				nn.Conv2d(ci, co, (1, 1))
			) for ci, co in zip(ch[:-2], ch[1:-1])
		])

		self.gc = nn.ModuleList([GatedConvolution(c) for c in ch[1:-1]])

		self.out1 = nn.Sequential(
			nn.Conv2d(ch[-2], ch[-1], (1, 1), bias=False),  # todo
			# nn.Sigmoid()  # nn.BCEWithLogitsLoss() is used to calculate loss, which itself uses sigmoid
		)

		self.scale = nn.UpsamplingBilinear2d(scale_factor=downscale_ss) if downscale_ss != 1 else None

		self.out2 = nn.Sequential(
			Concatenation(),
			nn.Conv2d(ch[-1] + 1, ch[-1], (1, 1), bias=False),
			nn.Sigmoid()

		)

	def forward(self, x: list, image_gradients):
		edge_out = self.conv_in[0](x[0])
		for idx, (x_, res, gc) in enumerate(zip(x[1:], self.res, self.gc)):
			edge_out = res(edge_out)
			y = self.conv_in[idx + 1](x_)
			edge_out = gc(edge_out, y)
		edge_out = self.out1(edge_out)
		if self.scale is not None:
			edge_out = self.scale(edge_out)
		x = self.out2([edge_out, image_gradients])
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
	"""
	Note, the code [1] implementation of the original MSRF-net [2] uses bias=True in both convolutions [3], this is
	removed since batch norm will remove the bias anyway. See also the original resnet paper [4] and batch norm [5].
	[1] https://arxiv.org/pdf/2105.07451v1.pdf
	[2] https://github.com/NoviceMAn-prog/MSRF-Net
	[3] https://github.com/NoviceMAn-prog/MSRF-Net/issues/21
	[4] https://arxiv.org/pdf/1512.03385v1.pdf
	[5] https://arxiv.org/pdf/1502.03167.pdf
	"""

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
		x = x + self.layers(x)
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
			nn.Conv2d(c + 1, c + 1, (1, 1)),  # todo: set bias=False and apply BN after?
			nn.ReLU(),
			nn.Conv2d(c + 1, 1, (1, 1), bias=False),
			nn.BatchNorm2d(1),
			nn.Sigmoid()
		)
		self.conv = nn.Conv2d(c, c, (1, 1), bias=False)

	def forward(self, x1, x2):
		x2 = self.gate([x1, x2])
		x1 = x1 * x2
		x1 = self.conv(x1)
		return x1


class Concatenation(torch.nn.Module):
	""" This class might as well be replaced by torch.cat, but by making a separate Module for this I find it makes
	the code more readable"""

	def __init__(self):
		super().__init__()

	@staticmethod  # todo: does this interfere with backward propagation?
	def forward(x: List):
		return torch.cat(x, dim=1)


class Decoder(nn.Module):
	def __init__(self, c: List, scales: List):
		""" c: output channels of MSRF subnetwork """
		super().__init__()
		if len(c) != 4:
			# raise ValueError(f'Current implementation only supports 4 inputs, not {len(c)}=')
			raise ValueError('Current implementation only supports 4 inputs, not %i', len(c))
		self.d2 = DecoderBlock(ci_dec=c[3], ci_msrf=c[2], scale=scales[3])
		self.deep_supervision_d2 = DeepSupervision(c[2], scale=scales[0] * scales[1] * scales[2])
		self.d3 = DecoderBlock(ci_dec=c[2], ci_msrf=c[1], scale=scales[2])
		self.deep_supervision_d3 = DeepSupervision(c[1], scale=scales[0] * scales[1])
		# though not mentioned in the original paper, their code does not use attention blocks in the last enc block
		self.d4 = DecoderBlock(ci_dec=c[1], ci_msrf=c[0], scale=scales[1], attentions=False)
		# todo: train network with/without final layer to see whether it has much impact
		self.final = nn.Sequential(
			nn.Conv2d(c[0], c[0], (3, 3), padding=(1, 1)),
			nn.ReLU(),
			nn.Conv2d(c[0], 1, (1, 1)),
			nn.Sigmoid()
		)

	def forward(self, x):
		x_dec = self.d2(x[3], x[2])  # n34 in original code
		ds1 = self.deep_supervision_d2(x_dec)  # deep-supervision
		x_dec = self.d3(x_dec, x[1])  # n24 in original code
		ds0 = self.deep_supervision_d3(x_dec)  # deep-supervision
		x = self.d4(x_dec, x[0])
		x = self.final(x)
		return x, ds0, ds1


class DecoderBlock(nn.Module):
	""" MSRF-Net https://arxiv.org/pdf/1907.05740v1.pdf  Fig.3
	Note, the figure contains a mistake, the multiplication sign after the S&E block should be a plus
	https://github.com/NoviceMAn-prog/MSRF-Net/issues/12
	"""

	def __init__(self, ci_dec, ci_msrf, scale: int, attentions: bool = True):
		super().__init__()
		self.dual_attention = DualAttention(ci_dec, ci_msrf) if attentions else None
		self.attention_gates = AttentionGates(ci_dec, ci_msrf) if attentions else None
		# transpose convolution below has kernel size (1,1) in original code
		#   https://github.com/NoviceMAn-prog/MSRF-Net/issues/23
		self.conv_t = nn.ConvTranspose2d(ci_dec, ci_msrf, (scale, scale), stride=(scale, scale))
		self.concatenation = Concatenation()
		c = 3 * ci_msrf if attentions else 2 * ci_msrf
		self.conv1 = nn.Conv2d(c, ci_msrf, (1, 1))  # todo
		self.res = ResBlock(ci_msrf)  # todo: original code uses a different layer sequence than resBlock

	def forward(self, x, x_msrf):
		if self.dual_attention is None:
			x = self.conv_t(x)
			x = self.concatenation([x_msrf, x])
		else:
			x = self.concatenation([self.dual_attention(x, x_msrf), self.attention_gates(x, x_msrf), self.conv_t(x)])
		x = self.conv1(x)
		x = self.res(x)
		return x


class DeepSupervision(nn.Module):
	def __init__(self, ci, scale):
		super().__init__()
		self.conv = nn.Conv2d(ci, 1, (1, 1))
		# self.sig = nn.Sigmoid()  # nn.BCEWithLogitsLoss() is used to calculate loss, which itself uses sigmoid
		self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

	def forward(self, x):
		x = self.conv(x)
		# x = self.sig(x)
		x = self.up(x)
		return x


class DualAttention(nn.Module):
	def __init__(self, ci_dec, ci_msrf):
		super().__init__()
		self.conv_t = nn.ConvTranspose2d(ci_dec, ci_msrf, (4, 4), stride=(2, 2), padding=(1, 1))
		self.concatenation = Concatenation()
		self.conv3 = nn.Conv2d(2 * ci_msrf, ci_msrf, (3, 3), padding=(1, 1))
		self.channel_attention = SqueezeExcite(ci_msrf)
		self.spatial_attention = SpatialAttention(ci_msrf, ci_msrf // 4)

	def forward(self, x, x_msrf):
		""" x: output previous decoder, x_msrf: output MSRF net """
		x = self.conv_t(x)
		x = self.concatenation([x, x_msrf])
		x = self.conv3(x)
		x = self.channel_attention(x) * self.spatial_attention(x)
		return x


class AttentionGates(nn.Module):
	""" Attention U-Net: Learning Where to Look for the Pancreas https://arxiv.org/pdf/1804.03999.pdf """

	def __init__(self, ci_dec, ci_msrf):
		super().__init__()
		self.conv1 = nn.Conv2d(ci_dec, ci_msrf, (1, 1))
		self.down = nn.Conv2d(ci_msrf, ci_msrf, (2, 2), stride=(2, 2))
		self.layers = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(ci_msrf, 1, (1, 1)),
			nn.Sigmoid(),
			nn.ConvTranspose2d(1, 1, (2, 2), stride=(2, 2)),  # todo: set bias=False and apply batch-norm instead?
		)
		self.bn = nn.BatchNorm2d(ci_msrf)  # not present in fig of paper

	def forward(self, x_dec, x_msrf):
		x = self.conv1(x_dec) + self.down(x_msrf)
		x = x_msrf * self.layers(x)
		x = self.bn(x)
		return x


class SpatialAttention(nn.Module):
	def __init__(self, ci, ch):
		super().__init__()
		self.layers = nn.Sequential(
			nn.Conv2d(ci, ch, (1, 1), bias=False),
			nn.BatchNorm2d(ch),
			nn.ReLU(),
			nn.Conv2d(ch, 1, (1, 1), bias=False),
			nn.BatchNorm2d(1),  # note: not present in original paper, todo: check effect with and without
			nn.Sigmoid(),
		)

	def forward(self, x):
		return self.layers(x)


def test():
	import cv2
	import numpy as np
	import pstats
	import cProfile

	img = torch.empty(size=(1, 1, 704, 520 + 8))
	img_grad = np.empty(img.shape)
	for idx, i in enumerate(img):
		img_grad[idx] = cv2.Canny(i.numpy().transpose((1, 2, 0)).astype(np.uint8), 10, 100)
	img_grad = torch.from_numpy(img_grad).to(torch.float32)

	# # encoder
	# c = [32, 64, 128, 256]
	# encoder = Encoder(ci=1, co=c)
	# torch.onnx.export(encoder, img, '../../temp/encoder.onnx')
	# out = encoder(img)
	# for o in out:
	# 	print(o.shape)
	#
	# # msrf subnet
	# msrf = MSRFSubNet(c)
	# out = msrf(out)
	# for o in out:
	# 	print(o.shape)
	#
	# # decoder
	# decoder = Decoder(c)
	# out_dec = decoder(out)
	# for o in out:
	# 	print(o.shape)
	#
	# # shape stream
	# shape_stream = ShapeStream(c, ch=[32, 32, 16, 8, 1])
	# out = shape_stream(out, img_grad)
	# for o in out:
	# 	print(o.shape)
	#
	# # final
	# final = Final(c, 1)
	# out = final(out_dec[0], out[0])
	# for o in out:
	# 	print(o.shape)

	m = MSRF(
		c_enc=[16, 64, 128, 256], c_ss=[16, 16, 16, 8, 1], downscales_enc=[2, 2, 2, 2], n_msrf_block_layers=2,
		downscale_ss=2
	)

	# memory footprint of model
	# mem_params = sum([param.nelement() * param.element_size() for param in m.parameters()])
	# mem_bufs = sum([buf.nelement() * buf.element_size() for buf in m.buffers()])
	# mem = mem_params + mem_bufs  # in bytes
	# Mb = 1024 ** 2
	# print(mem / Mb)
	# exit()

	# timing template
	out = m(img, img_grad)
	# with cProfile.Profile() as pr:
	# 	out = m(img, img_grad)
	# stats = pstats.Stats(pr)
	# stats.sort_stats(pstats.SortKey.TIME)
	# stats.print_stats()
	# stats.dump_stats(filename='../../temp/timing_model.prof')

	# out = m(img, img_grad)
	for o in out:
		print(o.shape)


if __name__ == "__main__":
	test()
