import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Train:
	def __init__(self, epochs, dl_train, dl_valid, model, device, optimizer, criterion, dtype):
		writer = SummaryWriter()
		model = model.to(device)
		for epoch in range(epochs):
			print('=' * 10 + ' epoch %3i ' % epoch + '=' * 10)
			self._train_epoch(model, dl_train, device, optimizer, criterion, writer, epoch)
			self._eval_epoch(model, dl_valid, device, criterion, writer, epoch)
		writer.close()

	@staticmethod
	def _train_epoch(model, dl_train, device, optimizer, criterion, writer, epoch):
		model.train()
		for idx, (img, canny, _, _, mask) in enumerate(dl_train):
			img, mask, canny = img.to(device), mask.to(device), canny.to(device)
			optimizer.zero_grad()
			y3, yb, y1, y2 = model(img, canny)
			loss = criterion((y3, yb, y1, y2), (mask, canny))
			loss.backward()
			optimizer.step()
			writer.add_scalar('loss train', loss.item(), epoch * len(dl_train) + idx)
			if idx == 0:
				writer.add_image('train_pred', y3[0, :, :, :], idx)
				writer.add_image('train_img', img[0, :, :, :], idx)
				writer.add_image('train_canny', canny[0, :, :, :], idx)
			print(f'Train\t{(idx + 1):3.0f}/{len(dl_train)}\t{loss.item():5.4f}')

	@staticmethod
	def _eval_epoch(model, dl_valid, device, criterion, writer, epoch):
		with torch.no_grad():
			model.eval()
			for idx, (img, canny, _, _, mask) in enumerate(dl_valid):
				img, mask, canny = img.to(device), mask.to(device), canny.to(device)
				y3, yb, y1, y2 = model(img, canny)
				loss = criterion((y3, yb, y1, y2), (mask, canny))
				writer.add_scalar('loss valid', loss.item(), epoch * len(dl_valid) + idx)
				print(f'Eval\t{(idx + 1):3.0f}/{len(dl_valid)}\t{loss.item():5.4f}')
			writer.add_image('test_pred', y3[0, :, :, :], epoch)
			writer.add_image('test_img', img[0, :, :, :], epoch)
			writer.add_image('test_canny', canny[0, :, :, :], epoch)
