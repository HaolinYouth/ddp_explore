"""
分布式训练极简体验 -- 01

初步体验pytorch中DistributeDataparallel的使用
模型 y = xA + b
"""

## 流程
# - 初始化pytorch分布式训练通信模块；
# - 创建模型（这里包括本地模型和分布式模型）
# - 创建损失函数和优化器
# - 计算（forward 和backward）和梯度更新
# - 多任务启动


import torch
import torch.nn as nn
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def run_worker(rank, world_size):
    
    # 通过环境变量MASTER_ADDR和MASTER_PORT设置rank0的IP和PORT信息，
    # rank0的作用相当于是协调节点，需要其他所有节点知道其访问地址;
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '7564'
    # 本例中后端选择的是nccl，通过设置NCCL_DEBUG环境变量为INFO，输出NCCL的调试信息
    os.environ['NCCL_DEBUG'] = 'INFO'

    # 创建一个默认的进程组
    # init_process_group：执行网络通信模块的初始化工作
    # backend：设置后端网络通信的实现库，可选的为gloo、nccl和mpi；
    # rank：为当前rank的index，用于标记当前是第几个rank，取值为0到work_size - 1之间的值；
    # world_size: 有多少个进程参与到分布式训练中;
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 创建本地的模型
    ## 将模型复制到GPU上 靠rank来标识GPU的id
    model = nn.Linear(10, 10).to(rank)

    # 构建分布式 DDP 模型
    ## 创建分布式模型 将local model复制到所有的副本上 并对数据进行切分
    ## 然后是的每个local model 都按照mini batch进行训练
    ddp_model = DDP(model, device_ids=[rank])

    # 定义损失
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=10e-3)

    # 前向过程
    ## 通过ddp_model 执行 forward 和 backward 计算 才能达到分布式计算的效果
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)

    # 反向传播
    loss_fn(outputs, labels).backward()

    # 更新参数
    optimizer.step()


def main():
    worker_size = 2
    mp.spawn(
        # run_worker: 是子进程执行的函数 会以fn(i, *args)的形式被调用
        # i 是process的id (0, 1, 2), *args为spawn的参数args
        run_worker,
        
        # 执行进程的参数 
        args=(worker_size,),

        # 进程的个数
        nprocs=worker_size,

        # 是否等待子进程执行完成
        join=True
    )

if __name__ == '__main__':
    main()