import torch
from torch import nn
from torchvision.models.vgg import vgg16
import torch.nn.functional as F # 确保导入 F

# GeneratorLoss 保持不变 (它不直接依赖判别器结构)
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        # 使用 VGG19 或 VGG16 的不同层可能效果不同，这里保持 VGG16
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval() # VGG16 relu5_2 feature
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_images, target_images):
        # 检查并复制通道数（如果输入是单通道）
        if out_images.size(1) == 1:
            out_images_3c = out_images.repeat(1, 3, 1, 1)
        else:
            out_images_3c = out_images
        if target_images.size(1) == 1:
            target_images_3c = target_images.repeat(1, 3, 1, 1)
        else:
            target_images_3c = target_images

        # 感知损失 (使用3通道输入)
        perception_loss = self.mse_loss(self.loss_network(out_images_3c), self.loss_network(target_images_3c))
        # 图像内容损失 (MSE, 使用原始通道数或3通道均可，这里用原始)
        image_loss = self.mse_loss(out_images, target_images)
        # TV 损失
        tv_loss = self.tv_loss(out_images) # TV Loss 通常在生成图像上计算

        # 组合损失，权重可能需要调整
        # 0.006 * perception_loss + 1.0 * image_loss + 2e-8 * tv_loss 是 SRGAN 论文中的常用权重
        # 你可以根据需要调整这些权重
        return image_loss + 0.006 * perception_loss + 2e-8 * tv_loss


# TVLoss 保持不变
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        # return t.size()[1] * t.size()[2] * t.size()[3]
        # 修正: t.numel() / t.size(0) 更通用
        if t.numel() == 0:
            return 0
        return t.numel() / t.size(0)


# 修改 compute_gradient_penalty 以适应新的判别器输入
def compute_gradient_penalty(D, real_hr, fake_sr, real_lr, device):
    """计算 WGAN-GP 的梯度惩罚
    Args:
        D: 判别器网络
        real_hr (Tensor): 真实的 HR 图像 (B, C, H, W)
        fake_sr (Tensor): 生成的 SR 图像 (B, C, H, W)
        real_lr (Tensor): 对应的 LR 图像 (B, C, H_lr, W_lr) 或 上采样后的 LR (B, C, H, W)
                         **注意:** 这里传入上采样后的 LR (与 real_hr/fake_sr 尺寸相同)
        device: 计算设备 ('cuda' or 'cpu')
    Returns:
        Tensor: 梯度惩罚值
    """
    batch_size, c, h, w = real_hr.shape
    alpha = torch.rand(batch_size, 1, 1, 1, device=device) # B, 1, 1, 1

    # 对 HR/SR 图像进行插值
    interpolated_hr_sr = (alpha * real_hr + (1 - alpha) * fake_sr).requires_grad_(True) # B, C, H, W

    # LR 图像对于这个插值样本是固定的，不需要插值，直接使用 real_lr (已上采样)
    # 但判别器需要两个输入
    d_interpolates = D(interpolated_hr_sr, real_lr) # B

    # 计算梯度
    fake_output = torch.ones_like(d_interpolates, device=device) # 创建与输出同形状的 1 张量

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolated_hr_sr, # 梯度是相对于插值的 HR/SR 图像计算的
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0] # 获取对应 inputs 的梯度

    # 计算梯度的 L2 范数并计算惩罚项
    gradients = gradients.view(batch_size, -1) # B, C*H*W
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty