import torch
from torch import nn
# from .carafe2 import CARAFE

try:
    from carafe import CARAFEPack   # https://github.com/myownskyW7/CARAFE official CUDA implementation
except ImportError:
    pass

class MNXBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, exp_r:int=4, kernel_size:int=7, do_res:int=True,):
        super().__init__()
        self.do_res = do_res
        
        # First convolution layer with DepthWise Convolutions
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, 
            kernel_size=kernel_size,  stride=1, padding=kernel_size//2, groups = in_channels)
        
        # Normalization Layer. GroupNorm
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        
        # Second convolution (Expansion) layer with Conv 1x1
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=exp_r*in_channels,
            kernel_size=1, stride=1, padding=0)
        
        # GeLU activations
        self.act = nn.GELU()
        
        # Third convolution (Compression) layer with Conv 1x1
        self.conv3 = nn.Conv2d(
            in_channels=exp_r*in_channels, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
            # print("res")
        return x1

class MNXDownBlock(MNXBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res = False)

        self.resample_do_res = do_res
        if do_res:
            self.res_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size = 1,stride = 2)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, stride=2, padding=kernel_size//2, groups=in_channels)
        
    def forward(self, x):
        x1 = super().forward(x)
        if self.resample_do_res:
            res = self.res_conv(x)
            x1 = x1 + res
            # print("resample res")

        return x1

class MNXUpBlock(MNXBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False)

        self.resample_do_res = do_res
        
        if do_res:            
            self.res_conv = nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        self.conv1 = nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = in_channels,
        )

    def forward(self, x):
        x1 = super().forward(x)
        # Asymmetry but necessary to match shape 不对称，但有必要匹配形状
        # 注：torch.nn.functional.pad 来对奇数3x3转置卷积核2x上采样导致的奇数特征图进行padding以保证
        x1 = torch.nn.functional.pad(x1, (1,0,1,0))
        
        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, (1,0,1,0))
            x1 = x1 + res

        return x1

# CARAFE上采样
class MNXUpBlockCARAFE(MNXBlock):
    def __init__(self, in_channels, out_channels, exp_r=4, kernel_size=7, do_res=False):
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False)

        self.resample_do_res = do_res
        
        if do_res:            
            self.res_conv = nn.ConvTranspose2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = 1,
                stride = 2
                )

        # self.carafe = CARAFE(in_channels)

        # https://github.com/myownskyW7/CARAFE 
        self.carafe = CARAFEPack(channels=in_channels, scale_factor=2)

    def forward(self, x):
        x1 = self.carafe(x)
        x1 = super().forward(x1)
        # Asymmetry but necessary to match shape 不对称，但有必要匹配形状
        # 注：torch.nn.functional.pad 来对奇数3x3转置卷积核2x上采样导致的奇数特征图进行padding以保证
        
        if self.resample_do_res:
            res = self.res_conv(x)
            res = torch.nn.functional.pad(res, (1,0,1,0))   # res路径使用了1x1转置卷积，也需要padding
            x1 = x1 + res

        return x1

class OutBlock(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        
        self.conv_out = nn.ConvTranspose2d(in_channels, n_classes, kernel_size=1)
        
    def forward(self, x): 
        return self.conv_out(x)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_channels = 32
    # net = MNXblock(n_channels, n_channels).to(device)
    # net = MNXDownBlock(n_channels, 2*n_channels, do_res=True).to(device)
    net = MNXUpBlock(n_channels, n_channels//2, do_res=True).to(device)
    
    # print(net)
    
    input_image = torch.randn(1, n_channels, 256, 256).to(device)
    sample_size = input_image.size()
    
    # summary(net, sample_size)
    
    output = net(input_image)
    print("输入形状:",input_image.shape)
    print("输出形状:",output.shape)