import torch
from .submission import batch_prediction_to_annotations


def test(ds, model, dst_pth_submission: str = 'submission.csv'):
	with torch.no_grad():
		model.eval()
		for idx, (img, canny, img_names) in enumerate(ds):
			pred = model(img, canny)
			pred = pred.detach()
			pred = pred.to(torch.device('cpu'))
			pred = torch.sigmoid(pred)
			pred = pred > 0.5
			batch_prediction_to_annotations(img_names, pred, dst_pth_submission)
