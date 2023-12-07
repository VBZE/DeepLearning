import torch
import os.path
from torch import optim
from Multi_Net import Multi_Sys
from DiceLoss import DiceLoss
from torch.utils.data import DataLoader
from Data_Process import MyTrainDataSet
from torchvision.utils import save_image

save_path = '/home/vbze/Image/MUNet/'
weight_path = '/home/vbze/Weight/MUNet/munet.pth'  # 定义权重地址
data_path = '/home/vbze/DataSet/Kvasir-Instrument/'  # 定义数据集地址
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 定义训练设备

if __name__ == '__main__':
    vit = Multi_Sys().to(device)
    data_loader = DataLoader(MyTrainDataSet(data_path), batch_size=20, shuffle=True, drop_last=True)
    if os.path.exists(weight_path):
        vit.load_state_dict(torch.load(weight_path))
        print('Successful load weight')
    else:
        print('Not successful load weight')

    epoch = 1  # 训练轮数
    stop_epoch = 10  # 停止次数
    learn_rate = 0.00005  # 学习率
    loss_fun = DiceLoss()  # 损失函数
    loss_fun = loss_fun.to(device)
    optimizer = optim.Adam(vit.parameters(), lr=learn_rate)  # 优化器

    while True:
        for i, (image, segment_image) in enumerate(data_loader):
            image, segment_image = image.to(device), segment_image.to(device)

            # 训练步骤开始
            out_image = vit(image)
            train_loss = loss_fun(out_image, segment_image)

            # 优化器优化模型
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # 每5次训练查看损失
            if i % 5 == 0:
                print(f'训练轮数：{epoch}-次数：{i}-训练损失：{train_loss.item()}')

            # 每50次训练保存权重
            if i % 50 == 0:
                torch.save(vit.state_dict(), weight_path)

            # 查看训练效果
            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]
            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')

        # 设置停止条件
        if epoch == stop_epoch:
            print('-------训练完成-------')
            break

        epoch = epoch + 1