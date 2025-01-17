#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import sleep
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def train():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    epoch = 20
    for i in range(epoch):
        print(f"[{os.getpid()}] (rank = {
              rank}, local_rank = {local_rank}) training...")
        model = ToyModel().cuda(local_rank)
        ddp_model = DDP(model, [local_rank])

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(local_rank))
        labels = torch.randn(20, 5).to(local_rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {
              local_rank}) loss = {loss.item()}\n")
        optimizer.step()
        sleep(1)


if __name__ == "__main__":
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    train()
    dist.destroy_process_group()
