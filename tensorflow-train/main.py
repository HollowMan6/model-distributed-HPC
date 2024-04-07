#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import tensorflow as tf
import mnist_setup
import slurm

per_worker_batch_size = 64
cluster, job_name, task_index = slurm.tf_config_from_slurm(0)
num_workers = len(cluster['worker'])

print(f"Job name: {job_name}")
print(f"Task index: {task_index}")
print(f"Num workers: {num_workers}")
print(f"TF_CONFIG: {os.environ['TF_CONFIG']}")

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
    # Model building/compiling need to be within `strategy.scope()`.
    multi_worker_model = mnist_setup.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
