import torch
import torch.nn as nn
import torch.optim as optim

from base_model import Model, OneStepTrainer


class ManualDataParallelTrainer(OneStepTrainer):
    def __init__(self, base_model: Model, criterion: nn.Module, learning_rate: float):
        """
        Add the base model instance and the criterion.
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
        """Find cuda device list
        TODO: compat layer
        """
        if not torch.cuda.is_available():
            return []
        device_count: int = torch.cuda.device_count()
        return [torch.device(f"cuda:{i}") for i in range(device_count)]

    @staticmethod
    def _device_copy(cpu_model: Model, device: torch.device) -> Model:
        """create the model copy on the target device
        as a general note:
        - in theory, we could use a different gpu for the source of truth:
        - eg, not use it for training etc.
        - but the cpu impl. is a nice canonical copy.
        also use state_dict as the trasnfer media
        - and, as a reminder, to never try to randomly initialize on each
        - available attached in parallel (at the start, too much chaos)
        """
        model = type(cpu_model)()
        model.load_state_dict(state_dict=cpu_model.state_dict())
        model.to(device=device)
        # optional verify copy, but slowdown
        return model

    def _create_replicas(
        self, base_model: Model, devices: List[torch.device]
    ) -> List[Model]:
        """Creates model replicas on the specified devices."""
        cpu_model: Model = base_model.cpu()
        # well the base always on CPU, no worry.
        replicas: List[Model] = [
            self._device_copy(cpu_model, device) for device in devices
        ]
        assert len(set(model.device for model in replicas)) == len(replicas), (
            "Model replicas are not on distinct devices."
        )
        return replicas

    def _create_optimizers(
        self, models: List[Model], learning_rate: float
    ) -> List[optim.Optimizer]:
        """Creates an SGD optimizer for each model replica.
        For terminology friendly: stochastic gradient descent.
        """
        return [
            optim.SGD(params=model.parameters(), lr=learning_rate) for model in models
        ]  # type: ignore

    def _chunk_data(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        split the data tensor into chunk for device,
        use input x,y on cpu,
        make sure batch size is divisible by the number of devices. TODO: handle this better
        """
        num_devices: int = len(self.models)
        assert x.size(dim=0) % num_devices == 0, (
            "Batch size must be divisible by the number of devices."
        )
        assert y.size(dim=0) % num_devices == 0, (
            "Batch size must be divisible by the number of devices."
        )

        x_chunks: Tuple[torch.Tensor, ...] = x.chunk(chunks=num_devices, dim=0)
        y_chunks: Tuple[torch.Tensor, ...] = y.chunk(chunks=num_devices, dim=0)
        return x_chunks, y_chunks

    def _average_gradients(self) -> None:
        """
        avg the graident on all model replicas,
        iterate thru all param of all replicas, collect grads,
        average on cpug, then distribute to each replica.
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
                    p.grad = avg_grad.to(dtype=p.device)
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
    from model_base import (
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
