import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from base_model import Model, OneStepTrainer


class TorchDataParallelTrainer(OneStepTrainer):
    """Manages a training step using torch.nn.DataParallel."""

    def __init__(
        self,
        base_model: nn.Module,
        criterion: nn.Module,
        device: Optional[torch.device | str] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        if not torch.cuda.is_available() and device.type == "cuda":
            raise RuntimeError("DataParallel requires CUDA, but it's not available.")

        # DataParallel handles model replication across GPUs
        # the other way to do it is to use torch.nn.parallel.DistributedDataParallel
        # (there are known issues with dataparallel):
        # TODO: update this to use DistributedDataParallel
        self.parallel_net = nn.DataParallel(base_model).to(device)
        self.device = next(self.parallel_net.parameters()).device
        self.criterion = criterion.to(self.device)
        # Create optimizer once
        self.optimizer = optim.SGD(self.parallel_net.parameters(), lr=0.01)

    def loss_and_backward_pass(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        if y.device != self.device:
            y = y.to(self.device)

        self.optimizer.zero_grad()
        # DataParallel handles scatter/gather internally
        output = self.parallel_net(x)
        loss = self.criterion(output, y)
        # DataParallel handles gradient reduction internally
        loss.backward()
        return loss

    def stepping_optimizer(self) -> None:
        self.optimizer.step()
