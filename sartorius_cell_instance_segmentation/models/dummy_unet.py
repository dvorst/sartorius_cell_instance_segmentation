"""
Stage   Resolution  Pool Padding
0       704 x 520   0
1       352 x 260   0
2       176 x 130   (0,1)
3       88 x  66    0
4       44 x  33    0
"""
import torch.nn as nn
import torch


def print_resolutions():
	w, h = 704, 520
	for i in range(5):
		if i == 2:
			h += 2
		print('%i %6.1f %6.1f % 8.1f' % (i, w, h, w * h))
		w /= 2
		h /= 2


class DummyUNet(nn.Module):
	def __init__(self):
		super().__init__()
		chs = (1, 64, 128, 256, 512, 1024)
		pad = (0, 0, (1, 0), 0)
		self.enc = Encoder(chs, pad)
		self.dec = Decoder(chs[::-1], pad[::-1])
		self.tail = nn.Conv2d(chs[2], 1, kernel_size=(1, 1))

	def forward(self, x, _):
		x = self.enc(x)
		x = self.dec(x)
		return x, torch.tensor([]), torch.tensor([]), torch.tensor([])


class Encoder(nn.Module):
	def __init__(self, chs, pad):
		super().__init__()
		self.blocks = nn.ModuleList([Block(ci, co) for ci, co in zip(chs[:-2], chs[1:-1])])
		self.pools = nn.ModuleList([nn.MaxPool2d(2, padding=p) for p in pad])
		self.bottleneck = Block(chs[-2], chs[-1])

	def forward(self, x):
		out = []
		for block, pool in zip(self.blocks, self.pools):
			x = block(x)
			out.append(x)
			x = pool(x)
		x = self.bottleneck(x)
		out.append(x)
		return out[::-1]


class Decoder(nn.Module):
	def __init__(self, chs, pad):
		super().__init__()
		self.ups, self.blocks = nn.ModuleList([]), nn.ModuleList([])
		for ci, co, p in zip(chs[:-2], chs[1:-1], pad):
			self.ups.append(nn.ConvTranspose2d(ci, co, kernel_size=(2, 2), stride=(2, 2), padding=p))
			self.blocks.append(Block(ci, co))
		self.final = nn.Sequential(
			Block(chs[-2], chs[-1]),
			torch.nn.Sigmoid()
		)

	def forward(self, enc_out):
		x = enc_out[0]
		for up, block, skip in zip(self.ups, self.blocks, enc_out[1:]):
			x = up(x)
			x = torch.cat([skip, x], dim=1)
			x = block(x)
		x = self.final(x)
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
