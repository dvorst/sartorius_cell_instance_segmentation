from abc import ABC
import torch
import torch.nn as nn
#
# class ResBlock(nn.Module):
# 	def __init__(self, ci, co):
# 		"""
# 		:param ci: input channels
# 		:param co: output channels
# 		"""
# 		super().__init__()
# 		self.conv1 = nn.Sequential(
# 			nn.BatchNorm2d(ci),
# 			nn.ReLU(),
# 			nn.Conv2d(ci, co, (1, 1), bias=False)
# 		)
# 		self.conv2 = nn.Sequential(
# 			nn.BatchNorm2d(co),
# 			nn.ReLU(),
# 			nn.Conv2d(co, co, (3, 3), padding=1, bias=False)
# 		)
# 		self.conv3 = nn.Sequential(
# 			nn.BatchNorm2d(co),
# 			nn.ReLU(),
# 			nn.Conv2d(co, co, (1, 1), bias=False)
# 		)
# 		self.identity = nn.Sequential(
# 			nn.Conv2d(ci, co, (1, 1), bias=False),
# 			nn.BatchNorm2d(co)
# 		)
#
# 	def forward(self, x):
# 		i = self.identity(x)
# 		x = self.conv1(x)
# 		x = self.conv2(x)
# 		x = self.conv3(x)
# 		x = i + x
# 		return x
#
#
# class BottleNeck(nn.Module):
# 	def __init__(self, ci, co):
# 		super().__init__()
#
#
# class Up(nn.Module):
# 	def __init__(self, ci, co):
# 		"""
# 		:param ci:
# 		:param co:
# 		"""
# 		super().__init__()
# 		self.
#
#
# class Unet(nn.Module):
# 	def __init__(self):
# 		super().__init__()
# 		c = [1, 16, 64, 128, 256]
# 		p = [(1, 0), (1, 0), (0, 2), (0, 0)]
# 		s = (4, 4)
#
# 		# Encoder
# 		# 1x502x704 = 353408
# 		self.enc_stage1 = ResBlock(c[0], c[1])
# 		self.down1 = nn.MaxPool2d(s, padding=p[0])
# 		# 16x126x176 = 354816
# 		self.enc_stage2 = ResBlock(c[1], c[2])
# 		self.down2 = nn.MaxPool2d(s, padding=p[1])
# 		# 64x32x44 = 90112
# 		self.enc_stage3 = ResBlock(c[2], c[3])
# 		self.down3 = nn.MaxPool2d(s, padding=p[2])
# 		# 128x12x8 = 12288
# 		self.enc_stage4 = ResBlock(c[3], c[4])
# 		self.down4 = nn.MaxPool2d(s, padding=p[3])
# 		# 256x3x2 = 1536
#
# 		# Bottleneck
# 		f = c[-1] * 3 * 2
# 		self.bottleneck = nn.Linear(f, f, bias=False)
#
# 		# Decoder
# 		#
# 		self.up1 = nn.ConvTranspose2d(
# 			c[-1], c[-1], s, s, output_padding=p[-1], bias=False
# 		)
# 		self.dec_stage1 = ResBlock(c[-1], c[-2])
# 		#
# 		self.up2 = nn.ConvTranspose2d(
# 			c[-2], c[-2], s, s, output_padding=p[-2], bias=False
# 		)
# 		self.dec_stage2 = ResBlock(c[-2], c[-3])
# 		#
# 		self.up3 = nn.ConvTranspose2d(
# 			c[-3], c[-3], s, s, output_padding=p[-3], bias=False
# 		)
# 		self.dec_stage3 = ResBlock(c[-3], c[-4])
# 		self.up2 = nn.ConvTranspose2d(
# 			c[-4], c[-4], s, s, output_padding=p[-4], bias=False
# 		)
# 		#
# 		self.up4 = nn.ConvTranspose2d(
# 			c[-5], c[-5], s, s, output_padding=p[-5], bias=False
# 		)
# 		self.dec_stage4 = nn.Sequential(
# 			nn.Sequential(
# 				nn.BatchNorm2d(ci),
# 				nn.ReLU(),
# 				nn.Conv2d(ci, co, (1, 1), bias=False)
# 			),
# 			 nn.Sequential(
# 				nn.BatchNorm2d(co),
# 				nn.ReLU(),
# 				nn.Conv2d(co, co, (3, 3), padding=1, bias=False)
# 			)
# 		)
#
