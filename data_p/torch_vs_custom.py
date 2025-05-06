import torch
import torch.nn as nn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
data_p_path = project_root / "data_p"
if str(data_p_path) not in sys.path:
    sys.path.insert(0, str(data_p_path))

from base_model import create_model, create_data
from torch_ref_dp import TorchDataParallelTrainer
from data_p import ManualDataParallelTrainer

DATA_SIZE = 400
CPU_DEVICE = torch.device("cpu")

print("Setting up models and data...")
base_model_torch = create_model().to(CPU_DEVICE)
base_model_manual = create_model().to(CPU_DEVICE)
base_model_manual.load_state_dict(base_model_torch.state_dict())

criterion = nn.MSELoss()

x_cpu, y_cpu = create_data(DATA_SIZE, CPU_DEVICE)
print("Setup complete.")

print("\nInstantiating trainers...")
try:
    torch_trainer = TorchDataParallelTrainer(
        base_model=base_model_torch, criterion=criterion
    )
    print(f"Using {torch_trainer.device} for Torch DP trainer.")
except RuntimeError as e:
    print(f"Error creating Torch DP Trainer (maybe no CUDA?): {e}")
    torch_trainer = None

try:
    manual_trainer = ManualDataParallelTrainer(
        base_model=base_model_manual, criterion=criterion
    )
    print(f"Using {manual_trainer.devices} for Manual DP trainer.")
except RuntimeError as e:
    print(f"Error creating Manual DP Trainer (requires CUDA): {e}")
    manual_trainer = None
except ValueError as e:
    print(f"Error creating Manual DP Trainer: {e}")
    manual_trainer = None

loss_torch = None
if torch_trainer:
    print("\nRunning TorchDataParallelTrainer step...")
    loss_torch = torch_trainer.loss_and_backward_pass(x_cpu, y_cpu)
    torch_trainer.stepping_optimizer()
    print(f"Torch DP Loss: {loss_torch.item()}")
    loss_torch = loss_torch.cpu()
else:
    print("\nSkipping TorchDataParallelTrainer step.")

loss_manual = None
models_manual = None
if manual_trainer:
    print("\nRunning ManualDataParallelTrainer step...")
    loss_manual = manual_trainer.loss_and_backward_pass(x_cpu, y_cpu)
    manual_trainer.stepping_optimizer()
    print(f"Manual DP Loss: {loss_manual.item()}")

    assert_models_same_weights(manual_trainer.models)
    print("Manual replicas verified to be synchronized.")
    models_manual = manual_trainer.models
    loss_manual = loss_manual.cpu()
else:
    print("\nSkipping ManualDataParallelTrainer step.")

print("\n--- Comparison Results ---")

if loss_torch is not None and loss_manual is not None:
    loss_diff = torch.abs(loss_torch - loss_manual)
    print("Loss Comparison:")
    print(f"  Torch DP Loss:  {loss_torch.item():.8f}")
    print(f"  Manual DP Loss: {loss_manual.item():.8f}")
    print(f"  Absolute Diff:  {loss_diff.item():.8g}")
else:
    print("Skipping loss comparison as one or both trainers failed.")

if torch_trainer and models_manual:
    print("\nWeight Comparison (Manual[0] vs TorchDP.module):")
    print_impl_diff(models_manual, torch_trainer.parallel_net)
else:
    print(
        "\nSkipping weight comparison as one or both trainers failed or were skipped."
    )

print("\nComparison script finished.")
