import os
from os.path import join
from PIL import Image

def is_image_file(filename):
    """检查文件是否为图像文件"""
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def resize_image(image, max_size=512):
    """将图像分辨率限制在 max_size x max_size 以内，保持宽高比"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image

def generate_lr_images(hr_dir, lr_dir, upscale_factor=4, max_size=512):
    """从 HR 图像生成 LR 图像，并替换原始 HR 图像为降低分辨率后的版本"""
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)

    for filename in os.listdir(hr_dir):
        if is_image_file(filename):
            hr_image_path = join(hr_dir, filename)
            lr_image_path = join(lr_dir, filename)

            # 打开 HR 图像并转换为灰度图像
            hr_image = Image.open(hr_image_path).convert('L')  # 转换为灰度图像

            # 将 HR 图像分辨率限制在 max_size x max_size 以内，并覆盖保存
            hr_image = resize_image(hr_image, max_size)
            hr_image.save(hr_image_path)  # 覆盖保存降低分辨率后的 HR 图像
            print(f"Resized HR image saved: {hr_image_path}")

            # 生成 LR 图像
            lr_width = hr_image.width // upscale_factor
            lr_height = hr_image.height // upscale_factor
            lr_image = hr_image.resize((lr_width, lr_height), Image.BICUBIC)

            # 保存 LR 图像
            lr_image.save(lr_image_path)
            print(f"Generated LR image: {lr_image_path}")

if __name__ == "__main__":
    # 设置路径和参数
    hr_dir = 'data/test/SRF_2/target'  # HR图像所在的文件夹
    lr_dir = 'data/test/SRF_2/data'  # 生成的LR图像保存的文件夹
    upscale_factor = 2  # 上采样因子
    max_size = 512  # HR 图像的最大分辨率

    # 生成 LR 图像并替换 HR 图像
    generate_lr_images(hr_dir, lr_dir, upscale_factor, max_size)