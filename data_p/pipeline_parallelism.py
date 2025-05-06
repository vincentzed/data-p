"""
Pipeline Parallelism GPipe-style.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict

from base_model import Model, OneStepTrainer


class PipelineStage(nn.Module):
    """Single pipeline stage."""

    def __init__(self, stage_id: int, sub_model: nn.Module, device: torch.device):
        super().__init__()
        self.stage_id = stage_id
        self.sub_model = sub_model.to(device)
        self.device = device
        # For storing input activation needed for backward pass
        self.stored_input_activation: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicitly move input to this stage's device
        x = x.to(self.device)
        # Store input activation for backward pass
        # detach() is important to prevent graph issues if x is from a previous stage
        # that requires_grad.
        self.stored_input_activation = x.detach().requires_grad_(True)
        return self.sub_model(self.stored_input_activation)

    def recompute_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recomputes forward pass."""
        x = x.to(self.device)
        input_for_grad = x.detach().requires_grad_(True)
        output = self.sub_model(input_for_grad)
        return output, input_for_grad


class PipelineParallelTrainer(OneStepTrainer):
    """Orchestrates pipeline parallelism."""

    def __init__(
        self,
        base_model: Model,
        num_stages: int,
        num_microbatches: int,
        devices: List[torch.device],
        criterion: nn.Module,
        use_rematerialization: bool = False,
    ):
        super().__init__()

        if len(devices) != num_stages:
            raise ValueError("Number of devices must match number of stages.")

        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.devices = devices
        self.criterion = criterion.to(devices[-1])  # Loss computed on last device
        self.use_rematerialization = use_rematerialization

        # Split model and create stages
        split_sub_models = self._split_model_specific(base_model, num_stages)
        self.stages: List[PipelineStage] = []
        for i in range(num_stages):
            stage = PipelineStage(
                stage_id=i, sub_model=split_sub_models[i], device=devices[i]
            )
            self.stages.append(stage)

        # Create optimizers, one for each stage
        self.optimizers: List[optim.Optimizer] = []
        for stage in self.stages:
            self.optimizers.append(optim.SGD(stage.sub_model.parameters(), lr=0.01))

        # Data structures for managing pipeline state per mini-batch
        self._clear_pipeline_storage()

    def _clear_pipeline_storage(self):
        # Activations: microbatch_idx -> stage_idx (output of stage) -> tensor
        self.activations: Dict[int, Dict[int, torch.Tensor]] = {}
        # Input Activations for backward (if not rematerializing or for remat input)
        self.input_activations_for_backward: Dict[int, Dict[int, torch.Tensor]] = {}
        # Gradients w.r.t. stage inputs: microbatch_idx -> stage_idx (input to stage) -> tensor
        self.input_gradients: Dict[int, Dict[int, torch.Tensor]] = {}

    def _split_model_specific(
        self, base_model: Model, num_stages: int
    ) -> List[nn.Module]:
        """Splits model by stages."""
        if num_stages == 1:
            return [base_model]
        if num_stages == 2:
            # Stage 0: fc1 -> relu
            # Stage 1: fc2
            stage0_model = nn.Sequential(base_model.fc1, base_model.relu)
            stage1_model = nn.Sequential(
                base_model.fc2
            )  # Wrap single layer for consistency
            return [stage0_model, stage1_model]
        elif num_stages == 3:  # Example for 3 stages
            stage0_model = nn.Sequential(base_model.fc1)
            stage1_model = nn.Sequential(base_model.relu)
            stage2_model = nn.Sequential(base_model.fc2)
            return [stage0_model, stage1_model, stage2_model]
        else:
            raise ValueError(
                f"Unsupported num_stages ({num_stages}) for this specific model split."
            )

    def _split_minibatch(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if x.size(0) % self.num_microbatches != 0:
            raise ValueError("Batch size must be divisible by num_microbatches.")
        microbatches_x = list(x.chunk(self.num_microbatches, dim=0))
        microbatches_y = list(y.chunk(self.num_microbatches, dim=0))
        return microbatches_x, microbatches_y

    def loss_and_backward_pass(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._clear_pipeline_storage()  # Reset for new mini-batch

        # Ensure optimizers are zero_grad before any forward/backward
        for opt in self.optimizers:
            opt.zero_grad()

        microbatches_x, microbatches_y = self._split_minibatch(x, y)

        # --- Pipelined Forward Pass ---
        # Schedule: F_ij is forward of microbatch i on stage j
        # Clock cycle t:
        # t=0: F_00
        # t=1: F_10, F_01
        # t=2: F_20, F_11, F_02
        # ...
        # Maximum number of active "diagonals"
        total_fwd_steps = self.num_microbatches + self.num_stages - 1

        for clock in range(total_fwd_steps):
            for mb_idx_in_batch in range(self.num_microbatches):
                stage_idx = clock - mb_idx_in_batch
                if 0 <= stage_idx < self.num_stages:
                    # This microbatch is active on this stage in this clock cycle
                    current_stage = self.stages[stage_idx]

                    if stage_idx == 0:  # First stage takes from original microbatch
                        input_act = microbatches_x[mb_idx_in_batch]
                    else:  # Subsequent stages take from previous stage's output
                        input_act = self.activations[mb_idx_in_batch][stage_idx - 1]

                    # Ensure input_act is on CPU before sending to potentially different device stage
                    input_act = input_act.cpu()

                    output_act = current_stage(
                        input_act
                    )  # Stage.forward handles .to(device) and stores input

                    if mb_idx_in_batch not in self.activations:
                        self.activations[mb_idx_in_batch] = {}
                    self.activations[mb_idx_in_batch][stage_idx] = output_act

                    if not self.use_rematerialization:
                        if mb_idx_in_batch not in self.input_activations_for_backward:
                            self.input_activations_for_backward[mb_idx_in_batch] = {}
                        # Store the input that *this stage* received (after .to(device))
                        if current_stage.stored_input_activation is not None:
                            self.input_activations_for_backward[mb_idx_in_batch][
                                stage_idx
                            ] = current_stage.stored_input_activation
                    else:  # For rematerialization, store original input to stage
                        if mb_idx_in_batch not in self.input_activations_for_backward:
                            self.input_activations_for_backward[mb_idx_in_batch] = {}
                        # Store input_act (before stage.forward modified it with requires_grad)
                        self.input_activations_for_backward[mb_idx_in_batch][
                            stage_idx
                        ] = input_act.to(current_stage.device)

        # --- Pipelined Backward Pass ---
        # Gradients for each microbatch are accumulated
        # Schedule: B_ij is backward of microbatch i on stage j
        # Starts after all forward passes for a microbatch are done.
        # GPipe typically processes backward passes in reverse order of microbatches: M-1, M-2, ..., 0
        total_loss = torch.tensor(
            0.0, device=self.devices[-1]
        )  # Accumulate loss on last device

        total_bwd_steps = self.num_microbatches + self.num_stages - 1

        for clock in range(total_bwd_steps):
            # Iterate microbatches in reverse as per GPipe for backward scheduling
            for mb_idx_in_batch_rev in range(self.num_microbatches):
                mb_idx = (
                    self.num_microbatches - 1 - mb_idx_in_batch_rev
                )  # Actual microbatch index

                # Stage index for backward, counting from last stage
                stage_idx_from_last = clock - mb_idx_in_batch_rev
                if 0 <= stage_idx_from_last < self.num_stages:
                    stage_idx = (
                        self.num_stages - 1 - stage_idx_from_last
                    )  # Actual stage index
                    current_stage = self.stages[stage_idx]

                    # Determine incoming gradient (dLoss / dOutput_of_this_stage)
                    if stage_idx == self.num_stages - 1:  # Last stage
                        # Get final output of this microbatch from forward pass
                        final_output_mb = self.activations[mb_idx][stage_idx].to(
                            self.devices[-1]
                        )
                        target_mb = microbatches_y[mb_idx].to(self.devices[-1])

                        loss_mb = self.criterion(final_output_mb, target_mb)
                        # To get grad w.r.t. final_output_mb for this microbatch
                        # The gradient of loss_mb w.r.t final_output_mb
                        # final_output_mb.backward() would sum grads if called repeatedly
                        # So we compute grad for this specific microbatch's output
                        grad_output_of_stage = torch.autograd.grad(
                            loss_mb, final_output_mb
                        )[0]
                        total_loss += (
                            loss_mb.detach()
                        )  # Accumulate detached loss for reporting
                    else:  # Intermediate stages
                        grad_output_of_stage = self.input_gradients[mb_idx][
                            stage_idx + 1
                        ]

                    # Ensure grad_output_of_stage is on CPU before sending to stage
                    grad_output_of_stage = grad_output_of_stage.cpu().to(
                        current_stage.device
                    )

                    # Get stored input activation for this stage and microbatch
                    if self.use_rematerialization:
                        # Recompute forward for this stage to get activations
                        original_input_to_stage = self.input_activations_for_backward[
                            mb_idx
                        ][stage_idx]
                        _, input_for_grad = current_stage.recompute_forward(
                            original_input_to_stage
                        )
                        output_act_for_bwd = self.activations[mb_idx][
                            stage_idx
                        ]  # The output that received grad_output_of_stage
                    else:
                        input_for_grad = self.input_activations_for_backward[mb_idx][
                            stage_idx
                        ]
                        output_act_for_bwd = self.activations[mb_idx][stage_idx]

                    if (
                        not input_for_grad.requires_grad
                    ):  # Should have been set in stage.forward
                        input_for_grad.requires_grad_(True)

                    # Perform backward for this stage and microbatch
                    # This accumulates gradients in current_stage.sub_model.parameters()
                    # And returns gradient w.r.t. input_for_grad
                    # output_act_for_bwd is the tensor whose grad is grad_output_of_stage
                    # input_for_grad is the tensor whose grad we want
                    current_grads = torch.autograd.grad(
                        outputs=output_act_for_bwd,
                        inputs=input_for_grad,
                        grad_outputs=grad_output_of_stage,
                        retain_graph=True,  # Needed as graph is used multiple times
                    )
                    grad_input_to_stage = current_grads[0]

                    if stage_idx > 0:  # Store for previous stage if not the first stage
                        if mb_idx not in self.input_gradients:
                            self.input_gradients[mb_idx] = {}
                        self.input_gradients[mb_idx][stage_idx] = grad_input_to_stage

        # Average loss over microbatches
        avg_loss = total_loss / self.num_microbatches
        return avg_loss

    def stepping_optimizer(self) -> None:
        # Gradients have been accumulated. Now apply them.
        for i, optimizer in enumerate(self.optimizers):
            # Average gradients by num_microbatches
            # This assumes autograd sums gradients; if it averages, this isn't needed.
            # For nn.Module, .backward() sums gradients.
            for param in self.stages[i].sub_model.parameters():
                if param.grad is not None:
                    param.grad.div_(self.num_microbatches)

            optimizer.step()
            optimizer.zero_grad()  # Zero grads for the next mini-batch
