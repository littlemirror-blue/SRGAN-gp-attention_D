import math
import torch
from torch import nn
import torch.nn.functional as F # 导入 F

# Generator 部分保持不变
class RWMAB(nn.Module):
    def __init__(self, channels):
        super(RWMAB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        attention = self.attention(out)
        out = out * attention
        return residual + out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.rwmab_blocks = nn.ModuleList([RWMAB(64) for _ in range(5)]) # 使用5个RWMAB块
        upsample_blocks = [UpsampleBlock(64, 2) for _ in range(upsample_block_num)]
        self.upsample = nn.Sequential(*upsample_blocks)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        res = x # 保留初始特征用于跳跃连接（如果需要的话，当前代码未使用）
        for block in self.rwmab_blocks:
            x = block(x)
        # 可以考虑添加一个从 res 到这里的连接: x = x + res
        x = self.upsample(x)
        x = self.final_conv(x)
        return (torch.tanh(x) + 1) / 2

# 修改后的 Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 定义卷积块辅助函数 (Conv + BatchNorm + LeakyReLU)
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # HR/SR 分支 (对应图上半部分)
        self.hr_branch = nn.Sequential(
            conv_block(1, 64, stride=1, use_bn=False), # c64s1, LeakyReLU
            conv_block(64, 64, stride=2, use_bn=True)  # c64s2, BatchNorm, LeakyReLU
            # 输出特征图大小: B x 64 x H/2 x W/2
        )

        # LR 分支 (对应图下半部分)
        self.lr_branch = nn.Sequential(
            conv_block(1, 64, stride=1, use_bn=False), # c64s1, LeakyReLU (图中是c64s1)
            conv_block(64, 128, stride=2, use_bn=True) # c128s2, BatchNorm, LeakyReLU (为了空间对齐，这里用s2，通道数为128)
            # 输出特征图大小: B x 128 x H/2 x W/2 (假设输入LR已上采样到H x W)
        )

        # 共享的卷积层 (对应图 Concatenation 之后的部分)
        # 输入通道数为 64 (hr) + 128 (lr) = 192
        self.shared_branch = nn.Sequential(
            # D1 对应的块 (调整通道数以匹配拼接后的192)
            conv_block(192, 128, stride=1, use_bn=True), # c128s1
            conv_block(128, 128, stride=2, use_bn=True), # c128s2 -> 输出 H/4 x W/4
            # D2 对应的块
            conv_block(128, 256, stride=1, use_bn=True), # c256s1
            conv_block(256, 256, stride=2, use_bn=True), # c256s2 -> 输出 H/8 x W/8
            # D3 对应的块
            conv_block(256, 512, stride=1, use_bn=True), # c512s1
            conv_block(512, 512, stride=2, use_bn=True), # c512s2 -> 输出 H/16 x W/16
            # D4 对应的块
            conv_block(512, 1024, stride=1, use_bn=True), # c1024s1
            conv_block(1024, 1024, stride=2, use_bn=True), # c1024s2 -> 输出 H/32 x W/32
        )

        # 分类器部分 (对应图最后的全连接层)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Flatten(),
            nn.Linear(1024, 100),    # Fully Connected (100)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)        # Fully Connected (1) -> 输出原始分数 (WGAN-GP不需要Sigmoid)
        )

    def forward(self, img_hr_sr, img_lr):
        # img_lr 应该已经在外部被上采样到和 img_hr_sr 相同的大小
        # 如果没有，需要在这里添加 F.interpolate
        # 例如: img_lr_upsampled = F.interpolate(img_lr, size=img_hr_sr.shape[2:], mode='bicubic', align_corners=False)
        # 下面的代码假定输入的 img_lr 已经是上采样过的

        hr_feat = self.hr_branch(img_hr_sr)
        lr_feat = self.lr_branch(img_lr)

        # 拼接特征图
        concat_feat = torch.cat((hr_feat, lr_feat), dim=1) # 在通道维度拼接

        # 通过共享层
        shared_feat = self.shared_branch(concat_feat)

        # 通过分类器得到最终输出
        out = self.classifier(shared_feat)

        batch_size = out.size(0)
        return out.view(batch_size) # 返回 Bx1 的分数