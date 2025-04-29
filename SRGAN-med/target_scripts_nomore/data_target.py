import os
from os.path import join
from PIL import Image
from torchvision.transforms import Resize, ToPILImage, ToTensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def generate_lr_images(hr_dir, lr_dir, upscale_factor):
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    for filename in os.listdir(hr_dir):
        if is_image_file(filename):
            hr_image_path = join(hr_dir, filename)
            lr_image_path = join(lr_dir, filename)

            # 打开HR图像
            hr_image = Image.open(hr_image_path).convert('L')  # 转换为灰度图像

            # 计算LR图像的尺寸
            w, h = hr_image.size
            lr_size = (w // upscale_factor, h // upscale_factor)

            # 生成LR图像
            lr_image = hr_image.resize(lr_size, Image.BICUBIC)

            # 保存LR图像
            lr_image.save(lr_image_path)
            print(f"Generated LR image: {lr_image_path}")

if __name__ == "__main__":
    # 设置路径和参数
    hr_dir = 'data/test/SRF_2/target'  # HR图像所在的文件夹
    lr_dir = 'data/test/SRF_2/data'  # 生成的LR图像保存的文件夹
    upscale_factor = 4  # 上采样因子

    # 生成LR图像
    generate_lr_images(hr_dir, lr_dir, upscale_factor)