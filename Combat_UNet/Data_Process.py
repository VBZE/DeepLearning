import os.path
from torch.utils.data import Dataset
from Image_Process import keep_image_size_open
from torchvision import transforms

# 转换图片数据结构
transform = transforms.Compose([
    transforms.ToTensor()])

# 训练数据集
class MyTrainDataSet(Dataset):
    def __init__(self, path):
        self.path = path  # 数据集文件夹路径
        self.file = open(os.path.join(path, "train.txt"))  # 打开train.txt文件

        # 得到训练图片的文件名
        self.name = list()
        for line in self.file:
            line = line.strip('\n')
            self.name.append(line)

    # 数据集数量
    def __len__(self):
        return len(self.name)

    # 获得图片路径
    def __getitem__(self, index):
        segment_name = self.name[index]
        # masks文件夹图片路径, xx.png
        segment_path = os.path.join(self.path, 'masks', segment_name + '.png')
        # images文件夹图片路径, xx.jpg
        image_path = os.path.join(self.path, 'images', segment_name + '.jpg')

        # 读取图片
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)

# 测试数据集
class MyTestDataSet(Dataset):
    def __init__(self, path):
        self.path = path  # 数据集文件夹路径
        self.file = open(os.path.join(path, "test.txt"))  # 打开test.txt文件

        # 得到训练图片的文件名
        self.name = list()
        for line in self.file:
            line = line.strip('\n')
            self.name.append(line)

    # 数据集数量
    def __len__(self):
        return len(self.name)

    # 获得图片路径
    def __getitem__(self, index):
        segment_name = self.name[index]
        # masks文件夹图片路径, xx.png
        segment_path = os.path.join(self.path, 'masks', segment_name, 'png')
        # images文件夹图片路径, xx.jpg
        image_path = os.path.join(self.path, 'images', 'jpg')

        # 读取图片
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image), transform(segment_image)
