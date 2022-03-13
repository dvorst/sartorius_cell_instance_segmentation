import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional
import PIL.Image
import cv2
from pathlib import Path


class TestDataset(torch.utils.data.Dataset):

	def __init__(
			self,
			dtype=torch.float,
			dir_pth: str = '../input/sartorius-cell-instance-segmentation/test',
			imgs_extension: str = 'png'
	):
		self.files = list(Path(dir_pth).glob('*.' + imgs_extension))
		self.dtype = dtype

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		file = self.files[idx]
		img = (torchvision.transforms.functional.pil_to_tensor(PIL.Image.open(file)) / 255).to(self.dtype)
		canny = self._canny(img=img, dtype=self.dtype)
		img_name = file.stem
		return img, canny, img_name

	@staticmethod
	def _canny(img, dtype):
		return torch.tensor(cv2.Canny(
			(img.numpy() * 255).astype(np.uint8).squeeze(), threshold1=10, threshold2=100
		) / 255., dtype=dtype).unsqueeze(0)
