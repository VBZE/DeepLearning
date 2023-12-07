import torch
from torch import nn

# 搭建神经网络
class VBZE(nn.Module):
    def __init__(self):
        super(VBZE, self).__init__()

        # 网络结构(分类问题)
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # 二维卷积
            nn.MaxPool2d(kernel_size=2),  # 最大池化
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),  # 将卷积展平
            nn.Linear(64*4*4, 64),  # 线性层
            nn.Linear(64, 10)  # 得到10个类别
        )

    def forward(self,img):
        img = self.model(img)
        return img

if __name__ == '__main__':

    # 测试网络模型
    vbze = VBZE()
    # batchsize, inchannel, height, width
    input = torch.ones((64, 3, 32, 32))
    output = vbze(input)
    print(output.shape)