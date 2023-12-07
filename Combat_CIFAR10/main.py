import torch.optim
import torchvision
from torch import nn
from Net_CIFAR10 import VBZE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="/home/vbze/DataSet/CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=False)  # 训练数据集
test_data = torchvision.datasets.CIFAR10(root="/home/vbze/DataSet/CIFAR10", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=False)  # 测试数据集

# 定义训练设备
device = torch.device("cuda")

train_data_size = len(train_data)  # 训练数据集长度
test_data_size = len(test_data)  # 测试数据集长度
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))

# 使用DataLoader加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 创建模型网络
vbze = VBZE()
vbze = vbze.to(device)

# 创建损失函数
loss_fun = nn.CrossEntropyLoss()
loss_fun = loss_fun.to(device)

# 定义优化器
learn_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(vbze.parameters(), lr=learn_rate)  # 模型参数，学习率

# 设置训练网络参数
epoch = 10  # 训练轮数
total_test_step = 0  # 记录测试次数
total_train_step = 0  # 记录训练次数

# 添加TensorBoard
writer = SummaryWriter("/home/vbze/TensorLogs")

for i in range(epoch):
    print("-------第{}轮训练-------".format(i+1))

    # 训练步骤开始
    vbze.train()  # 只对Dropout层，BatchNorm层等作用
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = vbze(imgs)  # 图片在网络中训练
        loss = loss_fun(outputs, targets)  # 与目标对比得到损失

        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数

        # 打印训练次数
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    vbze.eval()  # 只对Dropout层，BatchNorm层等作用
    total_test_loss = 0  # 总的损失
    total_accuracy = 0  # 总的正确率
    with torch.no_grad():  # 取消更新参数
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = vbze(imgs)
            loss = loss_fun(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)表示横向比较
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存每一轮训练的结果
    torch.save(vbze, "/home/vbze/Modules/CIFAR10_Module/vbze_{}.pth".format(i))
    print("-------模型已保存{}-------".format(i))

writer.close()