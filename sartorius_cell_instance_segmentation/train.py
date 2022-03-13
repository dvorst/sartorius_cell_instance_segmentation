import torch
import matplotlib.pyplot as plt
from .summary_writer import SummaryWriterExtended
from .util import imshow
import torchvision.transforms.functional


def train(
		epochs, dl_train, dl_valid, model, device, optimizer, criterion, summary_writer: SummaryWriterExtended, dtype
):
	model = model.to(device)
	for epoch in range(epochs):
		print('=' * 10 + ' epoch %3i ' % epoch + '=' * 10)
		_train_epoch(model, dl_train, device, optimizer, criterion, summary_writer, epoch)
		_eval_epoch(model, dl_valid, device, criterion, summary_writer, epoch)
	summary_writer.close()


def _train_epoch(model, dl_train, device, optimizer, criterion, writer, epoch):
	batch_size = dl_train.batch_size

	model.train()
	for idx, (img, canny, _, _, mask, name) in enumerate(dl_train):
		img, mask, canny = img.to(device), mask.to(device), canny.to(device)
		optimizer.zero_grad()
		pred = model(img, canny)
		loss = criterion(pred, (mask, canny))
		loss.backward()
		optimizer.step()
		pred = pred[0] if isinstance(pred, list) else pred
		writer.add_scalar('loss train', loss.item(), epoch * len(dl_train) + idx)
		# print(f'Train\t{(idx + 1):3.0f}/{len(dl_train)}\t{loss.item():5.4f}')
		print('Train\t%3.0f/%i\t%5.4f' % (idx + 1, len(dl_train), loss.item()))
		# if idx < writer.n_imgs_per_epoch:
		# 	for idx_batch in range(batch_size):
		# 		# writer.append_images('train_canny', canny, epoch)
		# 		writer.append_images('train_img', img, epoch)
		# 		writer.append_images('train_mask', mask, epoch)
		# 		writer.append_images('train_pred', pred, epoch)
		pass
	# imshow(prediction_to_image(pred[0]))
	# imshow(img[0])
	# imshow(mask[0])
	# plt.show()
	show_overlay(img, mask, pred)


def _eval_epoch(model, dl_valid, device, criterion, writer, epoch):
	with torch.no_grad():
		model.eval()
		for idx, (img, canny, _, _, mask, name) in enumerate(dl_valid):
			img, mask, canny = img.to(device), mask.to(device), canny.to(device)
			pred = model(img, canny)
			loss = criterion(pred, (mask, canny))
			pred = pred[0] if isinstance(pred, list) else pred
			writer.add_scalar('loss valid', loss.item(), epoch * len(dl_valid) + idx)
			# print(f'Eval\t{(idx + 1):3.0f}/{len(dl_valid)}\t{loss.item():5.4f}')
			print('Eval\t%3.0f/%i\t%5.4f' % (idx + 1, len(dl_valid), loss.item()))
			# writer.append_images('test_canny', canny, epoch)
			# writer.append_images('test_img', img, epoch)
			# writer.append_images('test_mask', mask, epoch)
			# writer.append_images('test_pred', pred, epoch)
			pass
		# imshow(prediction_to_image(pred[0]))
		# imshow(img[0])
		# imshow(mask[0])
		# plt.show()
		show_overlay(img, mask, pred)


def prediction_to_image(prediction: torch.tensor):
	with torch.no_grad():
		img = torch.nn.Sigmoid()(prediction[0].detach())
		img = img * 255
		img = img.type(torch.uint8)
		img = torchvision.transforms.functional.to_pil_image(img)
		return img


def show_overlay(
		img, mask, pred, alpha=(0.5, 0.25, 0.25),
		col_img=(1., 1., 1.), col_mask=(0., 1., 0.),
		col_pred=(1., 0., 0.),
):
	with torch.no_grad():
		""" overlay mask on image """
		pred = torch.nn.Sigmoid()(pred[0].detach())

		# verify that sum of all three alpha channels is 1
		if not (0.999999999999 < sum(alpha) < 1.000000000001):
			raise ValueError(f'sum of alpha must be 1 but is {sum(alpha)}')

		# convert img/mask to colored images
		img = _gray_img_to_col(img[0], col_img)
		mask = _gray_img_to_col(mask[0], col_mask)
		pred = _gray_img_to_col(pred, col_pred)

		# create img
		img = tensor_to_pil(
			img * alpha[0] +
			mask * alpha[1] +
			pred * alpha[2]
		)

		# display img
		plt.figure(dpi=200)
		plt.imshow(img)
		plt.gca().set_axis_off()
		plt.tight_layout()
		plt.show()


def _gray_img_to_col(gray_img: torch.tensor, color):
	return torch.cat([gray_img * c for c in color], dim=0)


def tensor_to_pil(t: torch.tensor):
	""" transform tensor with values between 0 and 1 to PIL image """
	return torchvision.transforms.functional.to_pil_image(
		(t * 255).type(torch.uint8)
	)
