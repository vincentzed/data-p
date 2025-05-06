import torch
import torch.nn as nn
from typing import List

from base_model import Model


def _ring_allreduce_sum_inplace(tensor_list: List[torch.Tensor]):
    """
    Performs Ring All-Reduce summation simulation IN-PLACE.
    A reference impl.
    TODO: Use
    """
    num_gpus = len(tensor_list)
    if num_gpus <= 1:
        return

    first_tensor = tensor_list[0]
    tensor_shape = first_tensor.shape  # unused
    tensor_numel = first_tensor.numel()
    tensor_dtype = first_tensor.dtype
    tensor_device = first_tensor.device

    if tensor_numel == 0:
        return

    if tensor_numel % num_gpus != 0:
        raise ValueError(
            "Tensor size must be divisible by num_gpus for this simulation."
        )

    chunk_size = tensor_numel // num_gpus

    # impl. scatter-reduce
    recv_buffer = torch.empty(chunk_size, dtype=tensor_dtype, device=tensor_device)

    for k in range(num_gpus - 1):
        for r in range(num_gpus):
            send_rank = r
            recv_rank = (r + 1) % num_gpus

            chunk_idx = (r - k + num_gpus) % num_gpus
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size

            send_chunk_view = tensor_list[send_rank].view(-1)[start_idx:end_idx]

            if recv_buffer.device != tensor_list[recv_rank].device:
                recv_buffer = recv_buffer.to(tensor_list[recv_rank].device)
            recv_buffer.copy_(send_chunk_view)

            local_chunk_view = tensor_list[recv_rank].view(-1)[start_idx:end_idx]
            local_chunk_view.add_(recv_buffer)

    # impl. all-gather
    for k in range(num_gpus - 1):
        for r in range(num_gpus):
            send_rank = r
            recv_rank = (r + 1) % num_gpus

            chunk_idx = (r - k + 1 + num_gpus) % num_gpus
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size

            send_chunk_view = tensor_list[send_rank].view(-1)[start_idx:end_idx]

            if recv_buffer.device != tensor_list[recv_rank].device:
                recv_buffer = recv_buffer.to(tensor_list[recv_rank].device)
            recv_buffer.copy_(send_chunk_view)

            local_chunk_view = tensor_list[recv_rank].view(-1)[start_idx:end_idx]
            local_chunk_view.copy_(recv_buffer)


class RingReduceGradientAverager:
    """averages gradients across model replicas using ring all-reduce (simulation)."""

    def __init__(self, models: List[Model]):
        if not models:
            raise ValueError("Model list cannot be empty.")
        self.models = models
        self.num_replicas = len(models)
        if self.num_replicas > 0:
            pass

    def average_gradients(self):
        """averages gradients across all replicas using ring reduce (simulation)."""
        if self.num_replicas <= 1:
            return

        # process each parameter across all replicas
        for params_tuple in zip(*[model.parameters() for model in self.models]):
            grads_list = [p.grad for p in params_tuple if p.grad is not None]

            if len(grads_list) != self.num_replicas:
                continue

            # sum gradients using ring all-reduce
            _ring_allreduce_sum_inplace(grads_list)

            # scale to get average
            avg_factor = 1.0 / self.num_replicas
            for grad in grads_list:
                grad.mul_(avg_factor)
