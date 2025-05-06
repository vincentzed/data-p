"""
Manual Data Parallelism Implementation.

This module implements manual data parallelism where a model is replicated across
multiple CUDA devices, and training data is split into chunks processed in parallel.
The gradients are manually averaged across replicas to ensure synchronized training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Any

from base_model import Model, OneStepTrainer


class ManualDataParallelTrainer(OneStepTrainer):
    """
    Manual implementation of data parallelism across multiple CUDA devices.
    
    This trainer replicates a model across available CUDA devices and splits
    the input batch into chunks, processing each chunk on a different device.
    After the forward and backward passes, gradients are averaged across all
    replicas to maintain synchronization.
    
    Key features:
    - Automatic detection of available CUDA devices
    - Model replication with synchronized initialization
    - Data chunking and distribution
    - Manual gradient averaging and synchronization
    """
    
    def __init__(self, base_model: Model, criterion: nn.Module, learning_rate: float):
        """
        Initialize the manual data parallel trainer.
        
        Args:
            base_model: The model to replicate across devices (should be on CPU)
            criterion: Loss function to use for training
            learning_rate: Learning rate for the SGD optimizers
            
        Raises:
            RuntimeError: If no CUDA devices are available
        """
        # super().__init__()

        self.devices: List[torch.device] = self._get_cuda_devices()
        if not self.devices:
            raise RuntimeError(
                "ManualDataParallelTrainer requires at least one CUDA device. Of course use multiple!"
            )

        print(f"Manual DP Trainer using devices: {self.devices}")
        self.criterion: nn.Module = criterion
        self.models: List[Model] = self._create_replicas(
            base_model=base_model, devices=self.devices
        )
        self.optimizers: List[optim.Optimizer] = self._create_optimizers(
            models=self.models, learning_rate=learning_rate
        )

    @staticmethod
    def _get_cuda_devices() -> List[torch.device]:
        """
        Discover and return all available CUDA devices.
        
        Returns:
            List of CUDA device objects. Empty list if no CUDA devices available.
            
        Note:
            This provides a compatibility layer for different CUDA configurations.
        """
        if not torch.cuda.is_available():
            return []
        device_count: int = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(device_count)]

    @staticmethod
    def _device_copy(cpu_model: Model, device: torch.device) -> Model:
        """
        Create a copy of the model on the specified device.
        
        Args:
            cpu_model: Source model on CPU
            device: Target device for the model copy
            
        Returns:
            Model replica on the target device with synchronized weights
            
        Design Notes:
            - Uses CPU model as the canonical source of truth
            - Uses state_dict for transfer to ensure proper weight copying
            - Avoids random initialization on each device (maintains consistency)
            - Could be extended to use a different GPU as source instead of CPU
        """
        model = type(cpu_model)()  # Create new instance of same model class
        model.load_state_dict(state_dict=cpu_model.state_dict())  # Copy weights
        model.to(device=device)  # Move to target device
        # Optional: Add verification of successful copy (trades speed for safety)
        return model

    def _create_replicas(
        self, base_model: Model, devices: List[torch.device]
    ) -> List[Model]:
        """
        Create model replicas on the specified devices.
        
        Args:
            base_model: The base model to replicate (moved to CPU first)
            devices: List of target devices for replicas
            
        Returns:
            List of model replicas, one per device
            
        Ensures:
            - All replicas are on distinct devices
            - Base model is safely moved to CPU before replication
        """
        cpu_model: Model = base_model.cpu()  # Ensure base model is on CPU
        # Create replicas on each target device
        replicas: List[Model] = [
            self._device_copy(cpu_model, device) for device in devices
        ]
        # Verify all replicas are on distinct devices (safety check)
        assert len(set(model.device for model in replicas)) == len(replicas), (
            "Model replicas are not on distinct devices."
        )
        return replicas

    def _create_optimizers(
        self, models: List[Model], learning_rate: float
    ) -> List[optim.Optimizer]:
        """
        Create SGD optimizers for each model replica.
        
        Args:
            models: List of model replicas
            learning_rate: Learning rate for all optimizers
            
        Returns:
            List of SGD optimizers, one per model replica
            
        Note:
            Uses Stochastic Gradient Descent (SGD) for simplicity and clarity.
            Each optimizer manages parameters for its corresponding model replica.
        """
        return [
            optim.SGD(params=model.parameters(), lr=learning_rate) for model in models
        ]

    def _chunk_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Split input tensors into chunks for parallel processing.
        
        Args:
            x: Input features tensor (should be on CPU)
            y: Target labels tensor (should be on CPU)
            
        Returns:
            Tuple of (x_chunks, y_chunks) where each chunk corresponds to a device
            
        Requirements:
            - Batch size must be evenly divisible by number of devices
            - Input tensors should be on CPU for efficient chunking
            
        TODO: Handle cases where batch size is not evenly divisible
        """
        num_devices: int = len(self.models)
        
        # Ensure batch size is evenly divisible by number of devices
        assert x.size(dim=0) % num_devices == 0, (
            "Batch size must be divisible by the number of devices."
        )
        assert y.size(dim=0) % num_devices == 0, (
            "Batch size must be divisible by the number of devices."
        )

        # Split tensors along batch dimension
        x_chunks: Tuple[torch.Tensor, ...] = x.chunk(chunks=num_devices, dim=0)
        y_chunks: Tuple[torch.Tensor, ...] = y.chunk(chunks=num_devices, dim=0)
        return x_chunks, y_chunks

    def _average_gradients(self) -> None:
        """
        Average gradients across all model replicas for synchronization.
        
        This is the core of manual data parallelism:
        1. Iterate through corresponding parameters from all model replicas
        2. Collect gradients from each replica and move to CPU
        3. Compute average gradient on CPU
        4. Distribute averaged gradient back to each replica
        
        Process:
        - Skip parameters that don't have gradients (e.g., those not requiring grad)
        - Use CPU for gradient averaging to avoid device memory issues
        - Ensure synchronized training across all replicas
        """
        # Iterate over corresponding parameters from all models simultaneously
        for params_on_devices in zip(*[model.parameters() for model in self.models]):
            # param that don't require graident, skip
            grads_to_average: List[Any] = [
                p.grad.cpu() for p in params_on_devices if p.grad is not None
            ]

            if not grads_to_average:
                continue

            # cpu grad calculation (avg)
            avg_grad: torch.Tensor = torch.stack(tensors=grads_to_average).mean(dim=0)

            # send back to each replica
            for p in params_on_devices:
                if p.grad is not None:
                    # overwrite local grad with avg grad, and send back to each replica
                    p.grad = avg_grad.to(device=p.device)
                    # move to right device

    def loss_and_backward_pass(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass, loss calculation, and backward pass across replicas.
        Perform fwd, pass, loss, backward. (on all replicas)
        return avg loss.
        """
        if x.device.type != "cpu" or y.device.type != "cpu":
            print(
                "Warning: Input tensors `x` and `y` should ideally be on CPU for efficient chunking in this manual implementation."
            )
            # could add logic to move them to CPU if needed, not much sense here.

        # zero grads
        # standard as practice.
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # distribute data and compute
        x_chunks, y_chunks = self._chunk_data(x, y)

        device_losses: List[torch.Tensor] = []
        device_outputs: List[torch.Tensor] = []

        for i, model in enumerate(iterable=self.models):
            x_i: torch.Tensor = x_chunks[i].to(device=model.device)
            y_i: torch.Tensor = y_chunks[i].to(device=model.device)

            # compute fwd on the replica
            output_i = model(x_i)
            device_outputs.append(output_i)  # Store if needed later

            # loss
            loss_i = self.criterion(output_i, y_i)
            device_losses.append(loss_i)

            # backward pass
            loss_i.backward()  # now model.parameters() has grads (local to the device)

        # gather losses to CPU, assume the different device case
        avg_loss: torch.Tensor = torch.stack(
            tensors=[loss.cpu() for loss in device_losses]
        ).mean()
        # avg scaclar loss, maybe better impl.
        return avg_loss

    def stepping_optimizer(self) -> None:
        """
        Averages the gradients and steps all optimizers synchronously.
        This should be called *after* loss_and_backward_pass.
        Average all gradient, and step all optimizer in a sync way.
        To call it after loss_and_backward_pass.
        """
        # sync grads.
        self._average_gradients()

        # now grad is synced, step all optimizer. (all optimizer are the same)
        for optimizer in self.optimizers:
            optimizer.step()


# --- Example Usage (begin GPT generation) ---
if __name__ == "__main__":
    from base_model import (
        Model,
        create_model,
        create_data,
        OneStepTrainer,
    )

    N_DATA = 400
    CPU_DEVICE = torch.device("cpu")

    base_net = create_model()  # Create on CPU first
    criterion = nn.MSELoss()
    x_cpu, y_cpu = create_data(N_DATA, CPU_DEVICE)  # Data on CPU

    try:
        manual_trainer = ManualDataParallelTrainer(
            base_model=base_net, criterion=criterion, learning_rate=0.01
        )

        print("Performing manual DP loss and backward pass...")
        avg_loss_manual = manual_trainer.loss_and_backward_pass(x_cpu, y_cpu)
        print(f"Manual DP Avg Loss: {avg_loss_manual.item()}")

        print("Performing manual DP optimizer step...")
        manual_trainer.stepping_optimizer()
        print("Optimizer step completed.")

    except RuntimeError as e:
        print(f"Could not run manual trainer: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
