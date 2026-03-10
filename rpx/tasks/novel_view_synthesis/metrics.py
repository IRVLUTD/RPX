import numpy as np
from PIL import Image

def compute_nvs_metrics(pred_img, gt_img):
    if isinstance(pred_img, Image.Image):
        pred_img = np.array(pred_img)
    if isinstance(gt_img, Image.Image):
        gt_img = np.array(gt_img)
    mse = np.mean((pred_img - gt_img) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    return {'psnr': psnr, 'mse': mse}
