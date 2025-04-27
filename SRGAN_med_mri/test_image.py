#=====test_image的读取路径是通过命令行参数--image_name指定(当前文件夹): python test_image.py --image_name test_image.png=====
import argparse
import time

import torch
from PIL import Image
# from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage

from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
# 放大修改为默认为8
parser.add_argument('--upscale_factor', default=8, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
# 默认修改为epoch_2_18.pth(G为生成器,D为判别器(只在训练时使用))
#parser.add_argument('--model_name', default='netG_epoch_2_18.pth', type=str, help='generator model epoch name')
# 默认修改为epoch_4_22.pth(G为生成器,D为判别器(只在训练时使用))
#parser.add_argument('--model_name', default='netG_epoch_4_22.pth', type=str, help='generator model epoch name')
# 尝试修改为epoch_8_98.pth(G为生成器,D为判别器(只在训练时使用))
parser.add_argument('--model_name', default='netG_epoch_8_98.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, weights_only=True))
else:
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage, weights_only=True))

image = Image.open(IMAGE_NAME)
with torch.no_grad():
    image = ToTensor()(image).unsqueeze(0)
if TEST_MODE:
    image = image.cuda()

start = time.perf_counter()
out = model(image)
elapsed = (time.perf_counter() - start)
print('cost ' + str(elapsed) + 's')

out_img = ToPILImage()(out[0].data.cpu())
out_img.save('out_srf_' + str(UPSCALE_FACTOR) + '_' + IMAGE_NAME)
