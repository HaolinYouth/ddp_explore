"""
多卡训练的实例
mnist实例
"""

import os
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

# 多卡计算的库
import torch.distributed as dist
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

# 
from datetime import datetime
import argparse


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu, args):
    # 同时要修改训练函数
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    #########################################################
    # 包装模型
    model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    #########################################################

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    
    #########################################################
    # 增加一个分数据的
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    #########################################################


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               #### shuffle 要改为false #### 
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # non_blocking 允许异步数据传输
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model_ddp(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item())
                   )
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    # 节点总数
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')

    # 每个节点的GPU总数 [每个节点的GPU数是一样的]
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    
    # 当前节点在所有节点当中的序号
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    # train(0, args)

    # 进程总数 
    args.world_size = args.gpus * args.nodes

    # 所有的进程需要知道进程0的Ip地址和端口 
    # 一般用0是master进程
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7564'
    # pytorch 提供了 mp.spawn来在同一个节点启动该节点的所有进程
    # 每个进程运行 train(i, args), 其中 i 从 0 到args.gpus-1
    mp.spawn(train, nprocs=args.gpus, args=(args, ))

if __name__ == '__main__':
    main()