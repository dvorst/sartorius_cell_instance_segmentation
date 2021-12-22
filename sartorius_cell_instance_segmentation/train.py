import torch
import matplotlib.pyplot as plt


class Train:
	def __init__(
			self, epochs, dl_train, dl_valid, model, device, optimizer,
			criterion, dtype
	):
		pass

		loss_train = torch.ones(len(dl_train) * epochs) * float('nan')
		acc_train = torch.ones(len(dl_train) * epochs) * float('nan')
		loss_valid = torch.ones(len(dl_valid) * epochs) * float('nan')
		acc_valid = torch.ones(len(dl_valid) * epochs) * float('nan')

		labels = torch.tensor(
			[[0, 0], [1, 0]]
		).expand(dl_train.batch_size, 2, 2)

		for epoch in range(epochs):

			print('=' * 10 + ' epoch %3i ' % epoch + '=' * 10)
			model.train()
			for idx, (img, bounds, touch, mask) in enumerate(dl_train):
				if device != 'cpu':
					img, mask = img.to(device), mask.to(device)
				print(img.shape)
				print(bounds.shape)
				optimizer.zero_grad()
				print(img.dtype)
				print(bounds.dtype)
				out = model(img, bounds)
				y3, yb, y1, y2 = out
				loss = criterion(y3, yb, y1, y2, mask, labels)
				loss.backward()
				optimizer.step()
				i = epoch * len(dl_train) + idx
				loss_train[i] = loss.item()
			print('%3i/%i    %5.4f' % (idx + 1, len(dl_train), loss.item()))

			print('-' * 31)

			with torch.no_grad():
				model.eval()
				for idx, (img, bounds, touch, mask) in enumerate(dl_valid):
					x, y = img, mask
					yp = model(x)
					loss = criterion(yp, y)
					i = epoch * len(dl_valid) + idx
					loss_valid[i] = loss.item()
					print(
						'%3i/%i    %5.4f' %
						(idx + 1, len(dl_valid), loss.item())
					)

			plt.figure()
			xt = torch.linspace(0, epochs, len(loss_train))
			xv = torch.linspace(0, epochs, len(loss_valid))
			plt.plot(xt, loss_train)
			plt.plot(xv, loss_valid)
			plt.figure()
			plt.plot(xt, acc_train)
			plt.plot(xv, acc_valid)
			plt.show()
