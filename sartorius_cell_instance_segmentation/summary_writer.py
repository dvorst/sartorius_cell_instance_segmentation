import torch
import torch.utils.tensorboard


class SummaryWriterExtended(torch.utils.tensorboard.SummaryWriter):
	def __init__(
			self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
			n_imgs_per_epoch: int = 10
	):
		super().__init__(
			log_dir=log_dir, comment=comment, purge_step=purge_step, max_queue=max_queue, flush_secs=flush_secs,
			filename_suffix=filename_suffix
		)
		self.n_imgs_per_epoch = n_imgs_per_epoch
		self._ctr_imgs = {}
		self._last_epoch = {}

	def append_images(self, tag, img_tensor, epoch: int):
		""" similar to add_images, it adds images for a certain epoch until self.n_imgs_per_epoch are added """
		# reset image counter if new epoch started and also when images are added to [tag] for the first time
		if tag not in self._last_epoch.keys() or self._last_epoch[tag] != epoch:
			self._ctr_imgs[tag], self._last_epoch[tag] = 0, epoch
		# append/add image until n_imgs are added
		batch_size = img_tensor.shape[0]
		for idx in range(batch_size):
			if self._ctr_imgs[tag] >= self.n_imgs_per_epoch:
				return
			step = epoch * self.n_imgs_per_epoch + self._ctr_imgs[tag]
			self.add_image(tag, img_tensor[idx, :, :, :], global_step=step)
			self._ctr_imgs[tag] += 1

