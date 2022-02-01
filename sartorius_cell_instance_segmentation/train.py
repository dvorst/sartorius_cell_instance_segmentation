import torch
import matplotlib.pyplot as plt
from .summary_writer import SummaryWriterExtended


class Train:
	def __init__(
			self, epochs, dl_train, dl_valid, model, device, optimizer, criterion,
			summary_writer: SummaryWriterExtended, dtype
	):
		model = model.to(device)
		for epoch in range(epochs):
			print('=' * 10 + ' epoch %3i ' % epoch + '=' * 10)
			self._train_epoch(model, dl_train, device, optimizer, criterion, summary_writer, epoch)
			self._eval_epoch(model, dl_valid, device, criterion, summary_writer, epoch)
		summary_writer.close()

	@staticmethod
	def _train_epoch(model, dl_train, device, optimizer, criterion, writer, epoch):
		batch_size = dl_train.batch_size

		model.train()
		for idx, (img, canny, _, _, mask) in enumerate(dl_train):
			img, mask, canny = img.to(device), mask.to(device), canny.to(device)
			optimizer.zero_grad()
			y3, yb, y1, y2 = model(img, canny)
			loss = criterion((y3, yb, y1, y2), (mask, canny))
			loss.backward()
			optimizer.step()
			writer.add_scalar('loss train', loss.item(), epoch * len(dl_train) + idx)
			print(f'Train\t{(idx + 1):3.0f}/{len(dl_train)}\t{loss.item():5.4f}')
			if idx < writer.n_imgs_per_epoch:
				for idx_batch in range(batch_size):
					# writer.append_images('train_canny', canny, epoch)
					writer.append_images('train_img', img, epoch)
					writer.append_images('train_mask', mask, epoch)
					writer.append_images('train_pred', y3, epoch)

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
				# writer.append_images('test_canny', canny, epoch)
				writer.append_images('test_img', img, epoch)
				writer.append_images('test_mask', mask, epoch)
				writer.append_images('test_pred', y3, epoch)
