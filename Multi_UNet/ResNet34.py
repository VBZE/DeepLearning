import torch
from torch import nn
from torch.nn.functional import relu
from timm.models.layers import to_2tuple

class PatchPart(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=56, patch_size=4):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

# 实现Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out = out + residual
        return relu(out)

# 实现ResNet34主模块
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        # 前几层图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(3, 3, 1, stride=2),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        # 重复的layer，分别有3,4,6,3个residual block
        self.layer1 = self.make_layer(3, 96, 3, stride=2)
        self.layer2 = self.make_layer(96, 192, 4, stride=2)
        self.layer3 = self.make_layer(192, 384, 6, stride=2)
        self.layer4 = self.make_layer(384, 768, 3, stride=2)

    # 构造layer，包含多个residual block
    @staticmethod
    def make_layer(in_channel, out_channel, block_num, stride=1):
        short_cut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride, bias=False),
            nn.BatchNorm2d(out_channel))
        layers = list()
        layers.append(ResidualBlock(in_channel, out_channel, stride, short_cut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):

        patch = PatchPart()

        x = self.pre(x)  # img:16, 3, 112, 112
        x = self.layer1(x)  # img: 16, 96, 56, 56

        x = patch(x)

        # x = self.layer2(x)  # img: 16, 192, 28, 28
        # print(x.shape)
        # x = self.layer3(x)  # img: 16, 384, 14, 14
        # print(x.shape)
        # x = self.layer4(x)  # img: 16, 768, 7, 7
        # print(x.shape)
        return x

if __name__ == '__main__':
    img = torch.randn(16, 3, 224, 224)
    resnet = ResNet34()
    img = resnet(img)
    print(img.shape)