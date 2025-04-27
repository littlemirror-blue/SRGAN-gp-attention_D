import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F # 导入 F 用于上采样
import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss, compute_gradient_penalty # 导入修改后的 GP
from model import Generator, Discriminator # 导入修改后的 Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models with MedSRGAN-style Discriminator')
parser.add_argument('--crop_size', default=96, type=int, help='training images crop size (HR)')
parser.add_argument('--upscale_factor', default=2, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')
parser.add_argument('--d_updates', default=5, type=int, help='Number of D updates per G update') # D更新次数
parser.add_argument('--gp_weight', default=10, type=float, help='Gradient penalty weight') # GP权重
parser.add_argument('--adversarial_weight', default=1e-3, type=float, help='Adversarial loss weight for G') # G对抗损失权重
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate for Adam')

if __name__ == '__main__':
    opt = parser.parse_args()

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    D_UPDATES = opt.d_updates
    GP_WEIGHT = opt.gp_weight
    ADVERSARIAL_WEIGHT = opt.adversarial_weight
    LEARNING_RATE = opt.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Upscale Factor: {UPSCALE_FACTOR}")
    print(f"Crop Size (HR): {CROP_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"D Updates per G update: {D_UPDATES}")
    print(f"Gradient Penalty Weight: {GP_WEIGHT}")
    print(f"Adversarial Weight for G: {ADVERSARIAL_WEIGHT}")
    print(f"Learning Rate: {LEARNING_RATE}")


    train_set = TrainDatasetFromFolder('data/VOC2012/train', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/VOC2012/val', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    netG = Generator(UPSCALE_FACTOR).to(device)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator().to(device) # 使用新的 Discriminator
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    generator_criterion = GeneratorLoss().to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)) # WGAN 常用 betas
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)) # WGAN 常用 betas

    results = {'d_loss': [], 'g_loss': [], 'd_score_real': [], 'd_score_fake': [], 'psnr': [], 'ssim': []} # 调整记录项

    # 创建必要的目录
    os.makedirs('epochs', exist_ok=True)
    os.makedirs('training_results/SRF_' + str(UPSCALE_FACTOR), exist_ok=True)
    os.makedirs('statistics', exist_ok=True)


    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score_real': 0, 'd_score_fake': 0}

        netG.train()
        netD.train()

        for data, target in train_bar:
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            # 将数据移到 GPU
            real_hr = target.float().to(device) # 真实 HR 图像
            lr_img = data.float().to(device)    # 真实 LR 图像

            # --- 训练判别器 D ---
            # D 每更新 D_UPDATES 次，G 才更新 1 次
            for _ in range(D_UPDATES):
                optimizerD.zero_grad()

                # 生成假的 SR 图像
                fake_sr = netG(lr_img).detach() # detach() 防止梯度传回 G

                # **关键：将 LR 图像上采样到 HR/SR 尺寸**
                # 这里使用 'bicubic' 插值，也可以尝试 'bilinear'
                lr_img_upsampled = F.interpolate(lr_img, size=real_hr.shape[2:], mode='bicubic', align_corners=False)

                # 计算真实图像对的判别器输出
                real_out = netD(real_hr, lr_img_upsampled).mean()
                # 计算虚假图像对的判别器输出
                fake_out = netD(fake_sr, lr_img_upsampled).mean()

                # 计算梯度惩罚
                gradient_penalty = compute_gradient_penalty(netD, real_hr, fake_sr, lr_img_upsampled, device)

                # WGAN-GP 判别器损失
                # D 的目标是最大化 real_out - fake_out - GP * weight
                # 因此 D 的损失是 fake_out - real_out + GP * weight
                d_loss = fake_out - real_out + GP_WEIGHT * gradient_penalty

                # 反向传播和优化
                d_loss.backward()
                optimizerD.step()

                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score_real'] += real_out.item() * batch_size
                running_results['d_score_fake'] += fake_out.item() * batch_size


            # --- 训练生成器 G ---
            optimizerG.zero_grad()

            # 生成假的 SR 图像 (这次不 detach，需要计算 G 的梯度)
            fake_sr_g = netG(lr_img)
            # 同样需要上采样的 LR (如果之前没保存可以重新计算)
            lr_img_upsampled_g = F.interpolate(lr_img, size=real_hr.shape[2:], mode='bicubic', align_corners=False)

            # 计算生成图像对的判别器输出
            fake_out_g = netD(fake_sr_g, lr_img_upsampled_g).mean()

            # 计算生成器的内容损失 (像素 + 感知 + TV)
            content_loss = generator_criterion(fake_sr_g, real_hr)

            # 计算生成器的对抗损失
            # G 的目标是最大化 fake_out_g (即让 D 认为假的是真的)
            # 所以对抗损失是 -fake_out_g
            adversarial_loss = -fake_out_g

            # 生成器总损失
            g_loss = content_loss + ADVERSARIAL_WEIGHT * adversarial_loss

            # 反向传播和优化
            g_loss.backward()
            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size

            # 更新 tqdm 进度条描述
            train_bar.set_description(desc=(
                f"[{epoch}/{NUM_EPOCHS}] "
                f"Loss_D: {running_results['d_loss'] / (running_results['batch_sizes'] * D_UPDATES):.4f} " # D loss 平均到每次更新
                f"Loss_G: {running_results['g_loss'] / running_results['batch_sizes']:.4f} "
                f"D(real): {running_results['d_score_real'] / (running_results['batch_sizes'] * D_UPDATES):.4f} " # D score 平均
                f"D(fake): {running_results['d_score_fake'] / (running_results['batch_sizes'] * D_UPDATES):.4f}" # D score 平均
            ))

        # --- Epoch 结束后的验证和保存 ---
        netG.eval()
        out_path = f'training_results/SRF_{UPSCALE_FACTOR}/'
        # if not os.path.exists(out_path): # 已在开头创建
        #     os.makedirs(out_path)

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc='Validating')
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr.float().to(device)
                hr = val_hr.float().to(device)
                sr = netG(lr)

                # 计算验证指标
                # Clamp sr to [0, 1] range before calculating metrics if necessary
                sr = torch.clamp(sr, 0.0, 1.0)
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                # Ensure ssim input tensors are on the same device and correct format (B, C, H, W)
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size

                # 计算总的 PSNR 和 SSIM
                valing_results['psnr'] = 10 * log10(1.0 / (valing_results['mse'] / valing_results['batch_sizes'])) # 假设图像归一化到 [0, 1]
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                val_bar.set_description(
                    desc=f'[Validating] PSNR: {valing_results["psnr"]:.4f} dB SSIM: {valing_results["ssim"]:.4f}'
                )

                # 收集图像用于保存 (确保在 CPU 上操作)
                val_hr_restore_cpu = val_hr_restore.squeeze(0).cpu()
                hr_cpu = hr.data.cpu().squeeze(0)
                sr_cpu = sr.data.cpu().squeeze(0)
                val_images.extend([display_transform()(val_hr_restore_cpu), # 可能需要调整 display_transform
                                   display_transform()(hr_cpu),
                                   display_transform()(sr_cpu)])

            # 保存验证图像网格
            if val_images: # 确保 val_images 不为空
                val_images = torch.stack(val_images)
                num_val_samples = len(val_loader)
                grid_size = min(num_val_samples * 3, 15) # 最多显示 15 张图 (5组 LR/HR/SR)
                val_images = torch.chunk(val_images, max(1, val_images.size(0) // grid_size)) # 分块防止显存不足

                val_save_bar = tqdm(val_images, desc='[Saving Validation Results]')
                index = 1
                for image_chunk in val_save_bar:
                    # 确保 image_chunk 包含 3 的倍数张图像
                    if image_chunk.size(0) % 3 != 0:
                        image_chunk = image_chunk[:-(image_chunk.size(0) % 3)] # 丢弃末尾不足一组的图像
                    if image_chunk.size(0) > 0:
                        image = utils.make_grid(image_chunk, nrow=3, padding=5)
                        utils.save_image(image, os.path.join(out_path, f'epoch_{epoch:03d}_index_{index:02d}.png'), padding=5)
                        index += 1
            else:
                print("No validation images generated.")


        # 保存模型参数
        torch.save(netG.state_dict(), f'epochs/netG_epoch_{UPSCALE_FACTOR}_{epoch:03d}.pth')
        torch.save(netD.state_dict(), f'epochs/netD_epoch_{UPSCALE_FACTOR}_{epoch:03d}.pth')

        # 记录统计数据
        results['d_loss'].append(running_results['d_loss'] / (running_results['batch_sizes'] * D_UPDATES))
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score_real'].append(running_results['d_score_real'] / (running_results['batch_sizes'] * D_UPDATES))
        results['d_score_fake'].append(running_results['d_score_fake'] / (running_results['batch_sizes'] * D_UPDATES))
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        # 每 10 个 epoch 保存一次统计 csv 文件
        if epoch % 10 == 0:
            out_stats_path = 'statistics/'
            # if not os.path.exists(out_stats_path): # 已在开头创建
            #     os.makedirs(out_stats_path)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'],
                      'Loss_G': results['g_loss'],
                      'Score_D_Real': results['d_score_real'],
                      'Score_D_Fake': results['d_score_fake'],
                      'PSNR': results['psnr'],
                      'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(os.path.join(out_stats_path, f'srf_{UPSCALE_FACTOR}_train_results_epoch_{epoch:03d}.csv'), index_label='Epoch')

    # 训练结束后保存最终的统计数据
    final_stats_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'Loss_D': results['d_loss'],
              'Loss_G': results['g_loss'],
              'Score_D_Real': results['d_score_real'],
              'Score_D_Fake': results['d_score_fake'],
              'PSNR': results['psnr'],
              'SSIM': results['ssim']},
        index=range(1, NUM_EPOCHS + 1))
    data_frame.to_csv(os.path.join(final_stats_path, f'srf_{UPSCALE_FACTOR}_train_results_final.csv'), index_label='Epoch')

    print("Training finished.")