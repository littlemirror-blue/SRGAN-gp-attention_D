import argparse
import os
from math import log10
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator

def extract_key(image_name):
    """从图像名称中提取患者编号，去掉前缀'1.3.6.1.4.1.14519.5.2.1.6279.6001.'"""
    prefix = "1.3.6.1.4.1.14519.5.2.1.6279.6001."
    slice_index = image_name.find("_slice_")
    if slice_index != -1:
        full_patient_id = image_name[:slice_index]  # 提取患者编号部分
        if full_patient_id.startswith(prefix):
            key = full_patient_id[len(prefix):]  # 去掉前缀
        else:
            key = full_patient_id  # 如果没有前缀，使用完整患者编号
    else:
        key = image_name  # 如果没有'_slice_'，使用整个名称
    return key

def main():
    parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
    # 放大修改为默认为2
    parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
    # 默认修改为epoch_2_98.pth(G为生成器,D为判别器(只在训练时使用))
    parser.add_argument('--model_name', default='netG_epoch_2_98.pth', type=str, help='generator model epoch name')
    # 默认修改为epoch_4_65.pth(G为生成器,D为判别器(只在训练时使用))
    #parser.add_argument('--model_name', default='netG_epoch_4_65.pth', type=str, help='generator model epoch name')
    # 尝试修改为epoch_8_41.pth(G为生成器,D为判别器(只在训练时使用))
    #parser.add_argument('--model_name', default='netG_epoch_8_41.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    # 动态初始化 results
    results = {}

    model = Generator(UPSCALE_FACTOR).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, weights_only=True))

    test_set = TestDatasetFromFolder('data/test', upscale_factor=UPSCALE_FACTOR)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

    out_path = 'benchmark_results/SRF_' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]
        with torch.no_grad():
            lr_image = lr_image
            hr_image = hr_image

        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()

        sr_image = model(lr_image)

        if hr_image.size() != sr_image.size():
            print(f"Adjusting sr_image size from {sr_image.size()} to {hr_image.size()}")
            sr_image = F.interpolate(sr_image, size=(hr_image.size(2), hr_image.size(3)), mode='bicubic', align_corners=False)

        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()

        # 使用患者编号和切片编号保存图像
        key = extract_key(image_name)
        slice_part = image_name.split('_slice_')[1] if '_slice_' in image_name else 'unknown'
        save_name = f"{out_path}{key}_slice_{slice_part.split('.')[0]}_psnr_{psnr:.4f}_ssim_{ssim:.4f}.png"
        test_images = torch.stack(
            [display_transform()(hr_restore_img.squeeze(0)), display_transform()(hr_image.data.cpu().squeeze(0)),
             display_transform()(sr_image.data.cpu().squeeze(0))])
        image = utils.make_grid(test_images, nrow=3, padding=5)
        utils.save_image(image, save_name, padding=5)

        # 动态添加结果
        prefix = key  # 使用新的患者编号作为键
        if prefix not in results:
            results[prefix] = {'psnr': [], 'ssim': []}
        results[prefix]['psnr'].append(psnr)
        results[prefix]['ssim'].append(ssim)

    out_path = 'statistics/'
    saved_results = {'psnr': [], 'ssim': []}
    for item in results.values():
        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        if (len(psnr) == 0) or (len(ssim) == 0):
            psnr = 'No data'
            ssim = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)

    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results.csv', index_label='DataSet')

if __name__ == '__main__':
    main()