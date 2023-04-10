import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
import numpy as np
import time
from torch.utils.data.distributed import DistributedSampler

# 获取数据集
def get_data_iter(batch_size, root='./classify-leaves'):
    '''
    返回训练集和测试集的迭代器以及数据分类的类别数
    '''

    # 读取整个临时数据集
    train_augs = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])  # 对输入图片从PIL转成numpy 并resize

    data_images = ImageFolder(root='./classify-leaves', transform=train_augs)
    # ImageFolder读取上file 这里 file/label(用label作为文件名)/image.jpg
    # data_images[0]为图片数据  data_images[1]为根据图片所在文件夹名称设置的label 因为本次数据label在csv文件中所以这个label都为上一层文件夹名称


    # panda读取csv文件
    train_csv = pd.read_csv('./classify-leaves/train.csv')
    # print(len(train_csv))#18353
    # print(train_csv.label)
    # 显示：

    # 获取某个元素的索引的方法
    # 这个class_to_num即作为类别号到类别名称的映射
    class_to_num = train_csv.label.unique()  # series.unique()为panda库中函数返回列表中出现的所有的元素(相当于相同元素去重)
    # print(np.where(class_to_num == 'maclura_pomifera')[0][0])
    # np.where()返回满足条件的索引class为 tuple 后面加上[0]class改为numpy  两个[0]可以返回序号值而不是数组

    # 将训练集的label对应成类别号
    train_csv['class_num'] = train_csv['label'].apply(lambda x: np.where(class_to_num == x)[0][0])


    # apply()遍历label中所有数据 并且从class_to_num序列中返回label中的x相同元素的索引值
    # print(train_csv)

    # 创建数据集
    class leaves_dataset(Dataset):  # 继承自torch.utils.data.Dataset
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, item):
            label = self.labels[item]
            data = self.images[item][0]  # 届时传入一个ImageFolder对象，需要取[0]获取数据，不要标签
            return data, label


    imgs = data_images
    labels = train_csv.class_num
    Leaf_dataset = leaves_dataset(images=imgs, labels=labels)  # 实例化leaves_dataset

    indices = range(len(labels))  # 标签中的长度
    Leaf_dataset_tosplit = torch.utils.data.Subset(Leaf_dataset, indices)
    # 数据集中前18353为训练数据,后面数据没有标签 数据和标签其实不等长，但是传入dataloder后会自动删掉后面多余数据  这里提前取出训练数据，提前整理更清楚一点


    # 随机拆分，设测试集比率ratio：
    def to_split(dataset, ratio=0.1):
        num = len(dataset)
        part1 = int(num * (1 - ratio))
        part2 = num - part1
        train_set, test_set = torch.utils.data.random_split(dataset, [part1, part2])
        return train_set, test_set


    train_set, test_set = to_split(Leaf_dataset_tosplit, ratio=0.01)  # 测试集比率设0.01

    train_iter = DataLoader(train_set, batch_size,sampler=DistributedSampler(train_set))
    test_iter = DataLoader(test_set, batch_size,sampler=DistributedSampler(test_set))

    return train_iter, test_iter, len(class_to_num)

# 测试集上评估准确率
def evaluate_accuracy(data_iter, net, device=None):
    """评估模型预测正确率"""
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就用net的device
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            # 测试集上做数据增强（normalize）
            # X = test_augs(X)
            if isinstance(net, torch.nn.Module):
                net.eval()  # 将模型net调成 评估模式，这会关闭dropout

                # 累加这一个batch数据中判断正确的个数
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()

                net.train()  # 将模型net调回 训练模式
            n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('save') #建立一个保存数据用的东西，save是输出的文件名


    net = net.to(device)
    print('training on ', device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            #训练时使用数据增强
            train_augs = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    ])
            X = train_augs(X)
            y = y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()


            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)

        #数据写入tensorboard
        writer.add_scalar(tag="loss/train", scalar_value=train_l_sum / batch_count,
                          global_step=epoch + 1)
        writer.add_scalar(tag="acc/train", scalar_value=train_acc_sum / n,global_step=epoch + 1)
        writer.add_scalar(tag="acc/test", scalar_value=test_acc,global_step=epoch + 1)

        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

