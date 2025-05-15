from torch import nn

from newarchs.UNetMNX import UNetMNX, UNetMNX_concat
from newarchs.UNetMNX_blocks import MNXUpBlockCARAFE


__all__ = ['UNet', 'NestedUNet',
            'UNetResidual', 'UNetWithSE','UNetWithSECBAM','UNetWithCBAM',
            'UNetwDS',
            'UNetMNXSmall', 'UNetMNXBase', 'UNetMNXMedium', 'UNetMNXLarge',
            'UNetMNX_CBAM_M', 'UNetMNX_EMA_M', 'UNetMNX_EMA_CBAM_M', 
            'UNetMNX_EMA_1_M', 'UNetMNX_EMA_2_M', 'UNetMNX_M_cat',
            'UNetMNX_carafe_M',
            'UNetMNX_EMA_carafe_M','UNetMNX_EMA_carafe_S','UNetMNX_EMA_carafe_B','UNetMNX_EMA_carafe_L',
            ]


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


# ===================================================================================
# 2024.9.27 添加UNetResidual测试版
# 在这个版本中，DoubleConvResidual 类包含了残差连接逻辑
# ===================================================================================

class DoubleConvResidual(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # Add a 1x1 convolution for dimensionality matching if necessary
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.double_conv(x)
        if self.residual is not None:
            identity = self.residual(x)
        out += identity
        out = self.relu(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvResidual(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvResidual(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvResidual(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetResidual(nn.Module):
    def __init__(self, num_classes, input_channels=3, bilinear=True, channels=[64, 128, 256, 512, 1024], **kwargs): # 1.[64, 128, 256, 512, 1024] 2.[32, 64, 128, 256, 512]
        super(UNetResidual, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConvResidual(input_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4] // factor)
        self.up1 = Up(channels[4], channels[3] // factor, bilinear)
        self.up2 = Up(channels[3], channels[2] // factor, bilinear)
        self.up3 = Up(channels[2], channels[1] // factor, bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# ===================================================================================
# 2024.9.27 添加UNetResidual基础上改写的UNetWithSE测试版
# 这个版本的 UNet 在每个 DoubleConvResidualSE 层中都加入了 SE 注意力机制
# ===================================================================================

class SELayer(nn.Module):
    """Squeeze-and-Excitation (SE) layer: re-calibrate channel-wise feature responses."""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class DoubleConvResidualSE(nn.Module):
    """(convolution => [BN] => ReLU) * 2 with residual connection and SE block"""

    def __init__(self, in_channels, out_channels, mid_channels=None, reduction=16):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        # Add a 1x1 convolution for dimensionality matching if necessary
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        self.se_layer = SELayer(out_channels, reduction=reduction)

    def forward(self, x):
        identity = x
        out = self.double_conv(x)
        out = self.se_layer(out)
        if self.residual is not None:
            identity = self.residual(x)
        out += identity
        out = self.relu(out)
        return out

class DownWithSE(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvResidualSE(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpWithSE(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvResidualSE(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvResidualSE(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetWithSE(nn.Module):
    def __init__(self, num_classes, input_channels=3, bilinear=True, channels=[64, 128, 256, 512, 1024], reduction=16, **kwargs):
        super(UNetWithSE, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConvResidualSE(input_channels, channels[0], reduction=reduction)
        self.down1 = DownWithSE(channels[0], channels[1])
        self.down2 = DownWithSE(channels[1], channels[2])
        self.down3 = DownWithSE(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = DownWithSE(channels[3], channels[4] // factor)
        self.up1 = UpWithSE(channels[4], channels[3] // factor, bilinear)
        self.up2 = UpWithSE(channels[3], channels[2] // factor, bilinear)
        self.up3 = UpWithSE(channels[2], channels[1] // factor, bilinear)
        self.up4 = UpWithSE(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# ===================================================================================
# 2024.9.27 添加UNetWithSE基础上改写的UNetWithSECBAM
# 在这个版本的 UNet 中，我们在每次跳跃连接之后添加了 CBAM 注意力模块
# ===================================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class UpWithSECBAM(nn.Module):
    """Upscaling then double conv with CBAM"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvResidualSE(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvResidualSE(in_channels, out_channels)

        self.cbam = CBAM(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x
    
class UNetWithSECBAM(nn.Module):
    def __init__(self, num_classes, input_channels=3, bilinear=True, channels=[64, 128, 256, 512, 1024], reduction=16, **kwargs):
        super(UNetWithSECBAM, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConvResidualSE(input_channels, channels[0], reduction=reduction)
        self.down1 = DownWithSE(channels[0], channels[1])
        self.down2 = DownWithSE(channels[1], channels[2])
        self.down3 = DownWithSE(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = DownWithSE(channels[3], channels[4] // factor)
        self.up1 = UpWithSECBAM(channels[4], channels[3] // factor, bilinear)
        self.up2 = UpWithSECBAM(channels[3], channels[2] // factor, bilinear)
        self.up3 = UpWithSECBAM(channels[2], channels[1] // factor, bilinear)
        self.up4 = UpWithSECBAM(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# ===================================================================================
# 2024.9.27 添加UNetWithSECBAM基础上改写的UNetWithCBAM
# 这个版本的 UNet 去除了UNetWithSECBAM的 SE 注意力机制，保持单纯UNetResidual+CBAM
# ===================================================================================

class UpWithCBAM(nn.Module):
    """Upscaling then double conv with CBAM"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvResidual(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvResidual(in_channels, out_channels)

        self.cbam = CBAM(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.cbam(x)
        return x

class UNetWithCBAM(nn.Module):
    def __init__(self, num_classes, input_channels=3, bilinear=True, channels=[64, 128, 256, 512, 1024], reduction=16, **kwargs):
        super(UNetWithCBAM, self).__init__()
        self.n_channels = input_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConvResidual(input_channels, channels[0])
        self.down1 = Down(channels[0], channels[1])
        self.down2 = Down(channels[1], channels[2])
        self.down3 = Down(channels[2], channels[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(channels[3], channels[4] // factor)
        self.up1 = UpWithCBAM(channels[4], channels[3] // factor, bilinear)
        self.up2 = UpWithCBAM(channels[3], channels[2] // factor, bilinear)
        self.up3 = UpWithCBAM(channels[2], channels[1] // factor, bilinear)
        self.up4 = UpWithCBAM(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
# ===================================================================================
# 2024.9.28 添加UNet With deep_supervision
# 这个版本的 UNet 支持深度监督，当使用深度监督时，所有的输出都会上采样到与最浅层输出相同的尺寸。
# 同时新增了train中deep_supervision_unet参数对应的损失函数计算：
    # 在计算损失函数时，lower weight at lower resolution
    # 取output1的loss*(1/16),
    # 取output2的loss*(1/8),
    # 取output3的loss*(1/4),
    # 取output4的loss*(1/2),
    # 取output5的loss, 最后把这几个loss相加
# ===================================================================================
class UNetwDS(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision_unet=True, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision_unet

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[4], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=1)
            self.final5 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

            self.dsup1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
            self.dsup2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.dsup3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.dsup4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        # L--------------------------------------------↓
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x4_0)  # x4_0[b,nb_filter[4](512),16, 16] 使用底层x4_0 作为监督1
            output2 = self.final2(x3_1)  # x3_1[b,nb_filter[3](256),32, 32] 1-4需要上采样到与output5相同大小
            output3 = self.final3(x2_2)  # x2_2[b,nb_filter[2](128),64, 64]
            output4 = self.final4(x1_3)  # x1_3[b,nb_filter[1](64),128,128]
            output5 = self.final5(x0_4)  # x0_4[b,nb_filter[1](32),256,256] 使用 x0_4 作为最终输出

            output1 = self.dsup1(output1)
            output2 = self.dsup2(output2)
            output3 = self.dsup3(output3)
            output4 = self.dsup4(output4)

            return [output1, output2, output3, output4, output5]

        else:
            output = self.final(x0_4)
            return output
        
# ===================================================================================
# 2024.9.29 添加UNetnn
# 这个版本的 UNet 支持深度监督，当使用深度监督时，所有的输出都会上采样到与最浅层输出相同的尺寸。
# 从第二层开始的每个nnBlock的第一个卷积层都将使用stride=2，从而实现下采样
# 我们将把nnBlock中的ReLU替换为LeakyReLU，并将BatchNorm2d替换为InstanceNorm2d
# 将解码阶段的上采样改为使用转置卷积 nn.ConvTranspose2d
# 下采样深度改为6次，直到1/64尺寸为止
    # 原始UNet为4次，1/16尺寸为止，channel最深为512
    # 特征图数量(channel)维持最深512不变，增加的两层特征图数量为512不变
    # deep_supervision仍从1/16尺寸的特征图输出，共5个output，增加的两层不输出
# ===================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvDropoutNormNonlin(nn.Module):
    """
    标准nnUNet基本卷积块，包含卷积、Dropout、归一化和非线性激活
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else None
        self.norm = nn.InstanceNorm2d(out_channels)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x


class ResidualBlock(nn.Module):
    """
    标准nnUNet的残差块实现
    """

    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.0):
        super().__init__()
        self.conv1 = ConvDropoutNormNonlin(in_channels, out_channels, stride, dropout_p)
        self.conv2 = ConvDropoutNormNonlin(out_channels, out_channels, 1, dropout_p)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class nnUNetEncoder(nn.Module):
    """
    nnUNet编码器部分
    """

    def __init__(self, input_channels, base_features, num_stages=5, dropout_p=0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.pool_ops = nn.ModuleList()
        self.feature_sizes = []

        # 输入层处理
        self.input_block = ResidualBlock(input_channels, base_features, dropout_p=dropout_p)
        self.feature_sizes.append(base_features)

        # 下采样阶段
        in_features = base_features
        for i in range(num_stages):
            out_features = min(in_features * 2, 320)  # nnUNet标准实现会限制最大特征数
            self.pool_ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.stages.append(ResidualBlock(in_features, out_features, dropout_p=dropout_p))
            self.feature_sizes.append(out_features)
            in_features = out_features

    def forward(self, x):
        skip_connections = []

        # 输入层处理
        x = self.input_block(x)
        skip_connections.append(x)

        # 下采样阶段
        for i in range(len(self.stages)):
            x = self.pool_ops[i](x)
            x = self.stages[i](x)
            if i < len(self.stages) - 1:  # 最后一层不需要作为跳跃连接
                skip_connections.append(x)

        return x, skip_connections


class nnUNetDecoder(nn.Module):
    """
    nnUNet解码器部分
    """

    def __init__(self, encoder_features, dropout_p=0.0):
        super().__init__()
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # 上采样阶段 - 使用编码器特征大小来确保通道一致性
        for i in range(len(encoder_features) - 2, -1, -1):
            current_features = encoder_features[i + 1]
            skip_features = encoder_features[i]

            # 上采样方式：转置卷积
            self.upsamples.append(nn.ConvTranspose2d(
                current_features, skip_features, kernel_size=2, stride=2
            ))

            # 合并后处理：连接后的通道数是当前层特征数+跳跃连接特征数
            self.stages.append(ResidualBlock(
                current_features + skip_features, skip_features, dropout_p=dropout_p
            ))

    def forward(self, x, skip_connections):
        for i in range(len(self.stages)):
            # 上采样
            x = self.upsamples[i](x)

            # 获取对应的跳跃连接
            skip = skip_connections[-(i + 2)]  # 反向索引

            # 处理大小不匹配情况
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=True)

            # 连接通道
            x = torch.cat([x, skip], dim=1)

            # 应用残差块
            x = self.stages[i](x)

        return x

class UNetMNXSmall(UNetMNX):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels=input_channels,
            n_channels=32,
            n_classes=num_classes,
            exp_r=2,
            kernel_size=kernel_size,
            deep_supervision=ds_unetmnx,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        )

class UNetMNXBase(UNetMNX):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size = kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

class UNetMNXMedium(UNetMNX):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNXLarge(UNetMNX):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3]
        )
        
# deep_supervision=deep_supervision 因为此时并没有UNetMNX_CBAM并没有实现，只是作为一个中转，后面的UNetMNX_CBAM_M才是具体实现
# 所以等具体实现的时候才决定要选择与模型对应的深度监督机制
# deep_supervision=ds_unetmnx
class UNetMNX_CBAM(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        self.CBAM0 = CBAM(n_channels)
        self.CBAM1 = CBAM(n_channels*2)
        self.CBAM2 = CBAM(n_channels*4)
        self.CBAM3 = CBAM(n_channels*8)
    
    def forward(self, x):
        x = self.stem(x)

        # Encoder
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        # Decoder
        x_up_3 = self.up_3(x)
        dec_x = self.CBAM3(x_res_3) + x_up_3    # CBAM3
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3             # 不知道del有啥影响，可以做实验对比一下，此处先保留

        x_up_2 = self.up_2(x)
        dec_x = self.CBAM2(x_res_2) + x_up_2    # CBAM2
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = self.CBAM1(x_res_1) + x_up_1    # CBAM1
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = self.CBAM0(x_res_0) + x_up_0    # CBAM0
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x
        
# https://github.com/YOLOonMe/EMA-attention-module
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class UNetMNX_EMA(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        ema_factor = 8
        self.enc_block_0.add_module(str(len(self.enc_block_0)), EMA(n_channels, factor=ema_factor))
        self.enc_block_1.add_module(str(len(self.enc_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.enc_block_2.add_module(str(len(self.enc_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.enc_block_3.add_module(str(len(self.enc_block_3)), EMA(n_channels*8, factor=ema_factor))
        
        self.bottleneck.add_module(str(len(self.bottleneck)), EMA(n_channels*16, factor=ema_factor))
        
        self.dec_block_3.add_module(str(len(self.dec_block_3)), EMA(n_channels*8, factor=ema_factor))
        self.dec_block_2.add_module(str(len(self.dec_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.dec_block_1.add_module(str(len(self.dec_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.dec_block_0.add_module(str(len(self.dec_block_0)), EMA(n_channels, factor=ema_factor))
    
class UNetMNX_EMA_CBAM(UNetMNX_CBAM):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        ema_factor = 8
        self.enc_block_0.add_module(str(len(self.enc_block_0)), EMA(n_channels, factor=ema_factor))
        self.enc_block_1.add_module(str(len(self.enc_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.enc_block_2.add_module(str(len(self.enc_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.enc_block_3.add_module(str(len(self.enc_block_3)), EMA(n_channels*8, factor=ema_factor))
        
        self.bottleneck.add_module(str(len(self.bottleneck)), EMA(n_channels*16, factor=ema_factor))
        
        self.dec_block_3.add_module(str(len(self.dec_block_3)), EMA(n_channels*8, factor=ema_factor))
        self.dec_block_2.add_module(str(len(self.dec_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.dec_block_1.add_module(str(len(self.dec_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.dec_block_0.add_module(str(len(self.dec_block_0)), EMA(n_channels, factor=ema_factor))

class UNetMNX_CBAM_M(UNetMNX_CBAM):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )
        
class UNetMNX_EMA_M(UNetMNX_EMA):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNX_EMA_CBAM_M(UNetMNX_EMA_CBAM):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

# 仅编码EMA
class UNetMNX_EMA_1(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        ema_factor = 8
        self.enc_block_0.add_module(str(len(self.enc_block_0)), EMA(n_channels, factor=ema_factor))
        self.enc_block_1.add_module(str(len(self.enc_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.enc_block_2.add_module(str(len(self.enc_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.enc_block_3.add_module(str(len(self.enc_block_3)), EMA(n_channels*8, factor=ema_factor))
        
        self.bottleneck.add_module(str(len(self.bottleneck)), EMA(n_channels*16, factor=ema_factor))

# 仅解码EMA
class UNetMNX_EMA_2(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        ema_factor = 8
        
        self.dec_block_3.add_module(str(len(self.dec_block_3)), EMA(n_channels*8, factor=ema_factor))
        self.dec_block_2.add_module(str(len(self.dec_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.dec_block_1.add_module(str(len(self.dec_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.dec_block_0.add_module(str(len(self.dec_block_0)), EMA(n_channels, factor=ema_factor))

class UNetMNX_EMA_1_M(UNetMNX_EMA_1):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNX_EMA_2_M(UNetMNX_EMA_2):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNX_M_cat(UNetMNX_concat):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )



class UNetMNX_carafe(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
                                                    # [3,3,9,3] in Swin as opposed to [2,2,2,2,2] in nn
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )
        
        self.up_3 = MNXUpBlockCARAFE(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_2 = MNXUpBlockCARAFE(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_1 = MNXUpBlockCARAFE(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_0 = MNXUpBlockCARAFE(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

class UNetMNX_carafe_M(UNetMNX_carafe):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNX_EMA_carafe(UNetMNX):
    def __init__(self, 
        in_channels: int, 
        n_channels: int,
        n_classes: int, 
        exp_r: int = 4,                             # Expansion ratio as in Swin Transformers
        kernel_size: int = 7,                       # Ofcourse can test kernel_size
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,             # Can be used to test deep supervision
        do_res: bool = False,                       # Can be used to individually test residual connection
        do_res_up_down: bool = False,               # Additional 'res' connection on up and down convs
        block_counts: list = [2,2,2,2,2,2,2,2,2],   # Can be used to test staging ratio: 
    ):
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            n_classes=n_classes,
            exp_r=exp_r,
            kernel_size=kernel_size,
            enc_kernel_size=enc_kernel_size,
            dec_kernel_size=dec_kernel_size,
            deep_supervision=deep_supervision,
            do_res=do_res,
            do_res_up_down=do_res_up_down,
            block_counts=block_counts
        )

        # ema only apply to encoder
        ema_factor = 8
        self.enc_block_0.add_module(str(len(self.enc_block_0)), EMA(n_channels, factor=ema_factor))
        self.enc_block_1.add_module(str(len(self.enc_block_1)), EMA(n_channels*2, factor=ema_factor))
        self.enc_block_2.add_module(str(len(self.enc_block_2)), EMA(n_channels*4, factor=ema_factor))
        self.enc_block_3.add_module(str(len(self.enc_block_3)), EMA(n_channels*8, factor=ema_factor))
        
        self.bottleneck.add_module(str(len(self.bottleneck)), EMA(n_channels*16, factor=ema_factor))

        # carafe only apply to decoder
        self.up_3 = MNXUpBlockCARAFE(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_2 = MNXUpBlockCARAFE(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_1 = MNXUpBlockCARAFE(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.up_0 = MNXUpBlockCARAFE(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )


class UNetMNX_EMA_carafe_M(UNetMNX_EMA_carafe):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,4,4,4,4,4,4,3]
        )

class UNetMNX_EMA_carafe_S(UNetMNX_EMA_carafe):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels=input_channels,
            n_channels=32,
            n_classes=num_classes,
            exp_r=2,
            kernel_size=kernel_size,
            deep_supervision=ds_unetmnx,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2]
        )

class UNetMNX_EMA_carafe_B(UNetMNX_EMA_carafe):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[2,3,4,4,4,4,4,3,2],       
            kernel_size = kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [2,2,2,2,2,2,2,2,2]
        )

class UNetMNX_EMA_carafe_L(UNetMNX_EMA_carafe):
    def __init__(self, num_classes, input_channels=3, ds_unetmnx=True, kernel_size=3, **kwargs):
        super().__init__(
            in_channels = input_channels, 
            n_channels = 32,
            n_classes = num_classes, 
            exp_r=[3,4,8,8,8,8,8,4,3],       
            kernel_size=kernel_size,         
            deep_supervision=ds_unetmnx,             
            do_res=True,                     
            do_res_up_down = True,
            block_counts = [3,4,8,8,8,8,8,4,3]
        )
