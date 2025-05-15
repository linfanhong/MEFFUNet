import torch
from torch import nn
from .UNetMNX_blocks import OutBlock, MNXBlock, MNXDownBlock, MNXUpBlock

class UNetMNX(nn.Module):
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
        super().__init__()

        self.do_ds = deep_supervision

        if kernel_size is not None:
            self.enc_kernel_size = kernel_size
            self.dec_kernel_size = kernel_size
        else:
            self.enc_kernel_size = enc_kernel_size
            self.dec_kernel_size = dec_kernel_size

        # Stem卷积    
        self.stem = nn.Conv2d(in_channels, n_channels, kernel_size=1)
        
        # 如果 exp_r 本身就是一个列表，那么就不需要进行转换,
        self.exp_r = exp_r
        if type(self.exp_r) == int:  # 如果 exp_r 是整数，exp_r 将变成一个block_counts长度的列表，每个元素都等于原来的 exp_r 值
            self.exp_r = [self.exp_r for i in range(len(block_counts))]   # 例如 exp_r = 4，block_counts = [2,2,2,2,2,2,2,2,2]，那么 exp_r = [4,4,4,4,4,4,4,4,4]
        
        # 方法：
        # “*[]" 是一个星号解包操作，
        # 当你有一个包含模块的列表，并希望将这些模块作为单独的参数传递给 nn.Sequential 构造函数时，就可以使用星号解包。
        # [expression for item in iterable] 列表解析式是一个语法，用于生成一个列表。

        # 使用：
        # [block() for _ in range(n)] 表示循环 n 次, 每次循环创建一个新的block 实例，并将它添加到列表中
        # 下面的 [...] 列表包含了 self.block_counts[0] 个MNXblock实例，
        # 然后通过 nn.Sequential(*[...]) 将这些块解包,包装进一个序列中
        
        self.enc_block_0 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels,
                out_channels=n_channels, 
                exp_r=self.exp_r[0], kernel_size=self.enc_kernel_size, do_res=do_res) 
            for _ in range(block_counts[0])]
        )

        self.down_0 = MNXDownBlock(
            in_channels=n_channels,
            out_channels=2*n_channels,
            exp_r=self.exp_r[1], kernel_size=self.enc_kernel_size, do_res=do_res_up_down
        )
        
        self.enc_block_1 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2, 
                exp_r=self.exp_r[1], kernel_size=self.enc_kernel_size, do_res=do_res)
            for _ in range(block_counts[1])]
        )
        
        self.down_1 = MNXDownBlock(
            in_channels=2*n_channels,
            out_channels=4*n_channels,
            exp_r=self.exp_r[2], kernel_size=self.enc_kernel_size, do_res=do_res_up_down
        )
        
        self.enc_block_2 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=self.exp_r[2], kernel_size=self.enc_kernel_size, do_res=do_res)
            for _ in range(block_counts[2])]
        )

        self.down_2 = MNXDownBlock(
            in_channels=4*n_channels,
            out_channels=8*n_channels,
            exp_r=self.exp_r[3], kernel_size=self.enc_kernel_size, do_res=do_res_up_down
        )

        self.enc_block_3 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*8, 
                out_channels=n_channels*8, 
                exp_r=self.exp_r[3], kernel_size=self.enc_kernel_size, do_res=do_res)
            for _ in range(block_counts[3])]
        )
        
        self.down_3 = MNXDownBlock(
            in_channels=8*n_channels,
            out_channels=16*n_channels,
            exp_r=self.exp_r[4], kernel_size=self.enc_kernel_size, do_res=do_res_up_down
        )
        
        self.bottleneck = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*16,
                out_channels=n_channels*16,
                exp_r=self.exp_r[4], kernel_size=self.dec_kernel_size, do_res=do_res)
            for _ in range(block_counts[4])]
        )
        
        self.up_3 = MNXUpBlock(
            in_channels=16*n_channels,
            out_channels=8*n_channels,
            exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.dec_block_3 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[5])]
        )

        self.up_2 = MNXUpBlock(
            in_channels=8*n_channels,
            out_channels=4*n_channels,
            exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.dec_block_2 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[6])]
        )
        
        self.up_1 = MNXUpBlock(
            in_channels=4*n_channels,
            out_channels=2*n_channels,
            exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.dec_block_1 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[7])]
        )
        
        self.up_0 = MNXUpBlock(
            in_channels=2*n_channels,
            out_channels=n_channels,
            exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=do_res_up_down
        )

        self.dec_block_0 = nn.Sequential(*[
            MNXBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[8])]
        )
        
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes)
        
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels*2, n_classes=n_classes)
            self.out_2 = OutBlock(in_channels=n_channels*4, n_classes=n_classes)
            self.out_3 = OutBlock(in_channels=n_channels*8, n_classes=n_classes)
            self.out_4 = OutBlock(in_channels=n_channels*16, n_classes=n_classes)

        self.block_counts = block_counts
        
    def forward(self, x):
        x = self.stem(x)

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

        x_up_3 = self.up_3(x)
        dec_x = x_res_3 + x_up_3 
        x = self.dec_block_3(dec_x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3             # 不知道del有啥影响，可以做实验对比一下，此处先保留

        x_up_2 = self.up_2(x)
        dec_x = x_res_2 + x_up_2 
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        dec_x = x_res_1 + x_up_1 
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        dec_x = x_res_0 + x_up_0 
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x


# 跳跃连接改为concat
class UNetMNX_concat(UNetMNX):
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
        # 跳跃连接改为concat
        self.dec_block_3 = nn.Sequential(
            MNXBlock(
                in_channels=n_channels*8*2, # concat两倍的channel输入
                out_channels=n_channels*8,
                exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=False), # concat后输入输出c不同不做res
            *[MNXBlock(
                in_channels=n_channels*8,
                out_channels=n_channels*8,
                exp_r=self.exp_r[5], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[5] - 1)]
        )

        self.dec_block_2 = nn.Sequential(
            MNXBlock(
                in_channels=n_channels*4*2,
                out_channels=n_channels*4,
                exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=False),
            *[MNXBlock(
                in_channels=n_channels*4,
                out_channels=n_channels*4,
                exp_r=self.exp_r[6], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[6] - 1)]
        )

        self.dec_block_1 = nn.Sequential(
            MNXBlock(
                in_channels=n_channels*2*2,
                out_channels=n_channels*2,
                exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=False),
            *[MNXBlock(
                in_channels=n_channels*2,
                out_channels=n_channels*2,
                exp_r=self.exp_r[7], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[7] - 1)]
        )

        self.dec_block_0 = nn.Sequential(
            MNXBlock(
                in_channels=n_channels*2,
                out_channels=n_channels,
                exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=False),
            *[MNXBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=self.exp_r[8], kernel_size=self.dec_kernel_size, do_res=do_res)
            for i in range(block_counts[8] - 1)]
        )
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
        
        # 跳跃连接res和up相加改为concat
        # dec_x = x_res_ + x_up_ 改为
        # dec_x = torch.cat([x_res_, x_up_], 1)
        
        # Decoder
        x_up_3 = self.up_3(x)
        # dec_x = x_res_3 + x_up_3 
        dec_x = torch.cat([x_res_3, x_up_3], 1)
        x = self.dec_block_3(dec_x)
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        del x_res_3, x_up_3

        x_up_2 = self.up_2(x)
        # dec_x = x_res_2 + x_up_2 
        dec_x = torch.cat([x_res_2, x_up_2], 1)
        x = self.dec_block_2(dec_x)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        del x_res_2, x_up_2

        x_up_1 = self.up_1(x)
        # dec_x = x_res_1 + x_up_1 
        dec_x = torch.cat([x_res_1, x_up_1], 1)
        x = self.dec_block_1(dec_x)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        del x_res_1, x_up_1

        x_up_0 = self.up_0(x)
        # dec_x = x_res_0 + x_up_0 
        dec_x = torch.cat([x_res_0, x_up_0], 1)
        x = self.dec_block_0(dec_x)
        del x_res_0, x_up_0, dec_x

        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else: 
            return x
        
    