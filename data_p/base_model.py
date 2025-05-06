import torch
from abc import ABC, abstractmethod


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = torch.nn.Linear(in_features=2, out_features=2)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(in_features=2, out_features=1)

    @property
    def device(self) -> torch.device:
        try:
            return next(super().parameters()).device
        except StopIteration:
            raise ValueError(
                "No parameters found in the model. Don't use CPU for training."
            )

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def create_data(
    N_data: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(N_data, 2).to(device)
    y = torch.randn(N_data, 1).to(device)
    return x, y


def create_model() -> Model:
    return Model()


class OneStepTrainer(ABC):
    @abstractmethod
    def loss_and_backward_pass(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        return calculated loss.
        """
        pass

    @abstractmethod
    def stepping_optimizer(self) -> None:
        pass
