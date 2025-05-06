import torch
import torch.nn as nn
import unittest
from typing import List, Tuple

# Make sure src is in path for imports
import sys
from pathlib import Path

project_root = (
    Path(__file__).resolve().parent.parent.parent
)  # Adjust if test file moves
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from base_model import create_model, create_data
from pipeline_parallel import (
    PipelineParallelTrainer,
    LEARNING_RATE,
)  # Assuming LEARNING_RATE in pipeline_parallel for now


# --- Helper: Non-Pipelined Trainer for Comparison ---
class NonPipelinedTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.device = device

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, nn.Module]:
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu(), self.model.cpu()


class TestPipelineParallelism(unittest.TestCase):
    def setUp(self):
        self.base_model_cpu = create_model()  # Model from model_base.py
        self.criterion = nn.MSELoss()
        self.data_size = 16  # Must be divisible by num_microbatches
        self.input_features = 2  # From Model definition
        self.num_microbatches_test = 4  # Example

        # Prepare devices (use CPU if no CUDA for testing simplicity)
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            self.devices_2stages = [torch.device("cuda:0"), torch.device("cuda:1")]
        else:
            self.devices_2stages = [torch.device("cpu"), torch.device("cpu")]
            print(
                "Warning: CUDA not available or <2 GPUs. Running pipeline tests on CPU."
            )

        self.x_cpu, self.y_cpu = create_data(self.data_size, torch.device("cpu"))

    def test_model_split_2_stages(self):
        trainer = PipelineParallelTrainer(
            base_model=create_model(),
            num_stages=2,
            num_microbatches=1,
            devices=self.devices_2stages,
            criterion=self.criterion,
        )
        self.assertEqual(len(trainer.stages), 2)
        # Add more specific checks about layers in each stage if needed
        # e.g. check if fc1 is in stage[0].sub_model and fc2 in stage[1].sub_model
        self.assertIsInstance(trainer.stages[0].sub_model[0], nn.Linear)  # fc1
        self.assertIsInstance(trainer.stages[0].sub_model[1], nn.ReLU)  # relu
        self.assertIsInstance(trainer.stages[1].sub_model[0], nn.Linear)  # fc2

    def _get_flat_params(self, model_or_stages: nn.Module | List) -> torch.Tensor:
        params = []
        if isinstance(model_or_stages, nn.Module):
            for param in model_or_stages.parameters():
                params.append(param.detach().cpu().view(-1))
        elif isinstance(model_or_stages, list):  # List of PipelineStage
            for stage in model_or_stages:
                for param in stage.sub_model.parameters():
                    params.append(param.detach().cpu().view(-1))
        return torch.cat(params)

    def test_pipeline_vs_non_pipeline_2stages(self):
        """
        Compare one step of pipeline training vs non-pipelined training.
        Losses and final parameters should be very close.
        """
        num_stages = 2

        # --- Non-Pipelined Baseline ---
        model_np = create_model()  # Fresh model
        # Ensure it starts with same weights as what pipeline trainer will use
        # (Pipeline trainer gets base_model_cpu which is fresh)

        # Use the first device for non-pipelined for simplicity
        non_pipe_device = self.devices_2stages[0]
        trainer_np = NonPipelinedTrainer(
            model_np, self.criterion, LEARNING_RATE, non_pipe_device
        )
        loss_np, model_np_final_cpu = trainer_np.train_step(self.x_cpu, self.y_cpu)
        params_np_final = self._get_flat_params(model_np_final_cpu)

        # --- Pipelined ---
        model_pp_base = create_model()  # Fresh model for pipeline
        # model_pp_base.load_state_dict(model_np.cpu().state_dict()) # Ensure identical start
        # model_np is already on CPU after train_step

        trainer_pp = PipelineParallelTrainer(
            base_model=model_pp_base,  # Pass the fresh base model
            num_stages=num_stages,
            num_microbatches=self.num_microbatches_test,
            devices=self.devices_2stages,
            criterion=self.criterion,
        )
        loss_pp = trainer_pp.loss_and_backward_pass(self.x_cpu, self.y_cpu)
        trainer_pp.stepping_optimizer()
        params_pp_final = self._get_flat_params(trainer_pp.stages)
        loss_pp_cpu = loss_pp.cpu()

        # --- Comparisons ---
        print(f"\nTest: Pipeline ({num_stages} stages) vs Non-Pipeline")
        print(f"  Non-Pipelined Loss: {loss_np.item():.6f}")
        print(f"  Pipelined Loss:     {loss_pp_cpu.item():.6f}")
        self.assertAlmostEqual(
            loss_np.item(),
            loss_pp_cpu.item(),
            places=5,
            msg="Losses should be very close",
        )

        self.assertTrue(
            torch.allclose(params_np_final, params_pp_final, atol=1e-5),
            msg="Final model parameters should be very close",
        )
        print("  Parameters comparison passed.")

    # TODO: Add test for rematerialization if/when implemented
    # def test_pipeline_with_rematerialization(self):
    #     # Similar to test_pipeline_vs_non_pipeline_2stages
    #     # but with trainer_pp initialized with use_rematerialization=True
    #     pass


if __name__ == "__main__":
    unittest.main()
