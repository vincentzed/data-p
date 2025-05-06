# Data Parallelism from Scratch

**Context: Scaling Neural Network Training**
Training large-scale neural networks necessitates leveraging multiple GPUs to overcome single-device memory and compute limitations. **Data Parallelism (DP)** is a fundamental technique for distributed training, enabling processing larger datasets or achieving faster convergence by distributing the workload.
This is the standard single-GPU training happens.
- Take the input data (some tensor).
- Compute the forward pass, calculate the loss given the truth labels, compute the backward pass.
- Optimizer step, update the model parameters (w - c*g(w))

Unfortunately, we can't scale single GPUs to massive scale. This is because of fundamental limits of silicon physics.
Similarly to distributed systems, we can take the data, model weights, etc. and split it to other compute devices.
Data parallelism: one subset of parallelism strategies in ML.

Fundmentally we do the following:
- Chooose one datapoint (subset of tensors), and compute the forward and backward pass.
- Calculate the gradient for each datapoint.
- Within all compute devices, communicate the gradients to (a master device), and average all `n` gradients.
- Then send back the averaged gradient to all compute devices, and update the model parameters. (And keep training)

This approach accomplishes the following:
- Each data parallel worker is bottlenecked by its own computation (assume the communication is free).
- Weights are updated on each device so the model is perfectly consistent.
- Train on all `n` datapoints each iteration

Unfortunately, data parallism often results in a flaw: communication = bottleneck.
Transferring data from some central device back to others is very slow (eg we are limited by PCIe bandwith, and between networked systems, the latency is on the order of milliseconds). Nominally on GPU the latency is (microseconds).

Batching:
- Batch size of `M`, so each device generates `M` gradients, and averages them before synchroinzation, we decrease the communication cost by `M` times (since we increase the duration of one computational step, but the communication is less frequent).
- But then the model is no longer the same ... (if it were a single GPU. Not necessarily a problem, but it is a source of confusion).

Parameter averaging
- Train `M` data parallel models on a subset of a dsata, then average the model parameters together.
- But this is flawed (you can perform the gradient descent and find different results. Not reccomended).

So then we need:
- Model parallelism or tensor parallelism. 

**Project Goal: Implementing Data Parallelism FROM SCRATCH**
This project implements data parallelism manually ("from scratch") to provide a clear understanding of its core components. We do the following:

1.  Explicitly code the steps involved:
    *   Model replication using a CPU master copy for guaranteed initial synchronization (`_device_copy`, `_create_replicas`).
    *   Manual data sharding (`_chunk_data`).
    *   Execution of local forward/backward passes.
    *   A **naive gradient averaging** strategy involving CPU aggregation (GPU -> CPU -> GPU). We show the basic synchronization requirement.
2.  Compare the numerical output (loss, final weights) of the manual implementation against PyTorch's standard `nn.DataParallel` to verify correctness.
3.  Implement the **Ring All-Reduce** algorithm (`_ring_allreduce_sum_inplace`) conceptually.
    *   Gradient synchronization is often the performance bottleneck in DP. Centralized averaging (like the naive CPU method) scales poorly. Ring All-Reduce is a bandwidth-optimal, decentralized algorithm used in libraries like NCCL to accelerate this step.
    *   Aim to illustrate the two core phases (Scatter-Reduce, All-Gather) and the ring-based communication pattern. It serves as an educational contrast to the naive approach, demonstrating *how* efficiency is gained structurally.


---