import torch
from torch import nn
from torch.nn.functional import interpolate

# 卷积块
class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            # padding_mode='reflect'保证整张图都有特征
            nn.Conv2d(in_channel, out_channel, 3, 1, 1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # 数据标准化
            nn.Dropout2d(0.3),  # 数据正则化
            nn.LeakyReLU(),  # 激活函数
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,
                      padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, img):
        return self.layer(img)

# 下采样(池化)
class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(kernel_size=2)

    def forward(self, img):
        return self.layer(img)

# 上采样(插值法)
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, 1, 1)  # //表示除后取整

    def forward(self, img, feature_map):
        imgup = interpolate(img, scale_factor=2, mode='nearest')
        imgout = self.layer(imgup)
        copy = torch.cat((imgout, feature_map), dim=1)  # 特征拼接
        return copy

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样过程
        self.convd1 = Conv_Block(3, 64)
        self.dsm1 = DownSample()
        self.convd2 = Conv_Block(64, 128)
        self.dsm2 = DownSample()
        self.convd3 = Conv_Block(128, 256)
        self.dsm3 = DownSample()
        self.convd4 = Conv_Block(256, 512)
        self.dsm4 = DownSample()
        self.convd5 = Conv_Block(512, 1024)

        # 上采样过程
        self.sps1 = UpSample(1024)
        self.convu1 = Conv_Block(1024, 512)
        self.sps2 = UpSample(512)
        self.convu2 = Conv_Block(512, 256)
        self.sps3 = UpSample(256)
        self.convu3 = Conv_Block(256, 128)
        self.sps4 = UpSample(128)
        self.convu4 = Conv_Block(128, 64)

        # 输出过程
        self.fin_out = nn.Conv2d(64, 3, 3, 1, 1)
        self.fin_pro = nn.Sigmoid()  # 用于将数据映射到0~1之间

    def forward(self, img):
        # 下采样过程
        R1 = self.convd1(img)
        R2 = self.convd2(self.dsm1(R1))
        R3 = self.convd3(self.dsm2(R2))
        R4 = self.convd4(self.dsm3(R3))
        R5 = self.convd5(self.dsm4(R4))

        # 上采样过程
        O1 = self.convu1(self.sps1(R5, R4))
        O2 = self.convu2(self.sps2(O1, R3))
        O3 = self.convu3(self.sps3(O2, R2))
        O4 = self.convu4(self.sps4(O3, R1))

        # 输出过程
        return self.fin_pro(self.fin_out(O4))



