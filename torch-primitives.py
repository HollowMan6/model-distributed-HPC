#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist


def run():
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.set_default_device("cuda:" + str(local_rank))

    # send & receive
    tensor = torch.zeros(1)
    if rank % 2 == 0:
        tensor += rank + 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=rank + 1)
        print("after send, Rank ", rank, " has data ", tensor[0])
        dist.recv(tensor=tensor, src=rank + 1)
        print("after recv, Rank ", rank, " has data ", tensor[0])
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=rank - 1)
        print("after recv, Rank ", rank, " has data ", tensor[0])
        tensor += rank
        dist.send(tensor=tensor, dst=rank - 1)
        print("after send, Rank ", rank, " has data ", tensor[0])

    # barrier
    # Synchronize all processes.
    dist.barrier()

    # broadcast
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before broadcast", " Rank ", rank, " has data ", tensor)
    dist.broadcast(tensor, src=0)
    print("after broadcast", " Rank ", rank, " has data ", tensor)

    # all_reduce
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before reudce", " Rank ", rank, " has data ", tensor)
    dist.all_reduce(
        tensor, op=dist.ReduceOp.SUM
    )  # SUM, PRODUCT, MIN, MAX, BAND, BOR, and BXOR
    print("after reudce", " Rank ", rank, " has data ", tensor)

    # reduce
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before reudce", " Rank ", rank, " has data ", tensor)
    dist.reduce(
        tensor,
        dst=3,
        op=dist.ReduceOp.SUM,
    )
    print("after reudce", " Rank ", rank, " has data ", tensor)

    # all_gather
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before gather", " Rank ", rank, " has data ", tensor)
    # create an empty list we will use to hold the gathered values
    gather_list = [torch.zeros(2, dtype=torch.int64)
                   for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    print("after gather", " Rank ", rank, " has data ", tensor)
    print("after gather", " Rank ", rank, " has gather list ", gather_list)

    # gather
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before gather", " Rank ", rank, " has data ", tensor)
    if rank == 0:
        gather_list = [torch.zeros(2, dtype=torch.int64)
                       for _ in range(world_size)]
        dist.gather(tensor, dst=0, gather_list=gather_list)
        print("after gather", " Rank ", rank, " has data ", tensor)
        print("gather_list:", gather_list)
    else:
        dist.gather(tensor, dst=0)
        print("after gather", " Rank ", rank, " has data ", tensor)

    # scatter
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    print("before scatter", " Rank ", rank, " has data ", tensor)
    if rank == 0:
        scatter_list = [torch.tensor([i, i]) for i in range(world_size)]
        print("scater list:", scatter_list)
        dist.scatter(tensor, src=0, scatter_list=scatter_list)
    else:
        dist.scatter(tensor, src=0)
    print("after scatter", " Rank ", rank, " has data ", tensor)

    # reduce_scatter
    tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
    scatter_list = [torch.tensor([i, i]) for i in range(world_size)]
    print("scater list:", scatter_list)
    print("before reduce_scatter", " Rank ", rank, " has data ", tensor)
    dist.reduce_scatter(tensor, scatter_list, op=dist.ReduceOp.SUM)
    print("after reduce_scatter", " Rank ", rank, " has data ", tensor)

    # all_to_all
    input = torch.arange(world_size) + rank * world_size
    input = list(input.chunk(world_size))
    print("before all_to_all", " Rank ", rank, " has input ", input)
    output = list(torch.empty(
        [world_size], dtype=torch.int64).chunk(world_size))
    # Scatters list of input tensors to all processes in a group and
    # return gathered list of tensors in output list.
    dist.all_to_all(output, input)
    # Same as:
    # scatter_list = input
    # gather_list  = output
    # for i in range(world_size):
    # dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)
    print("after all_to_all", " Rank ", rank, " has output ", output)
    print("Rank ", rank, " finished")


if __name__ == "__main__":
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    run()
    dist.destroy_process_group()
