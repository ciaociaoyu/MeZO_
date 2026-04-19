import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "medium_models"))

from src.trainer import Trainer  # noqa: E402


class TinyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 3, bias=False),
            nn.Tanh(),
            nn.Linear(3, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class DirectionalProbeConsistencyTest(unittest.TestCase):
    def _build_trainer(self, *, efficient_zero_order: bool, zo_quantization_bits: int = 16):
        torch.manual_seed(0)
        trainer = Trainer.__new__(Trainer)
        trainer.args = SimpleNamespace(
            optimize_acc=False,
            n_gpu=1,
            seed=17,
            use_c_scale=False,
            zero_order_use_trainer_optim=False,
            zo_variant=None,
            change_grad_estimate=False,
            sparse_ratio=0.5,
            sparse_mask_strategy="percentile_per_layer",
            sparse_scope="trainable_only",
            sparse_mask_refresh_steps=100,
            sparse_log_active_fraction=False,
            zo_quantization_bits=zo_quantization_bits,
            zo_two_point_precision="fp16",
            zero_order_eps=1e-3,
            init_h=1e-3,
            enable_two_point_h_estimation=True,
            h_estimation_active_source="two_point",
            efficient_zero_order=efficient_zero_order,
            non_diff=False,
        )
        trainer.state = SimpleNamespace(global_step=0, zo_forward_step=0)
        trainer.cs = {}
        trainer._sparse_refresh_masks = None
        trainer._sparse_refresh_masks_step = None
        trainer._sparse_refresh_stats = None
        trainer._sparse_thresholds = None
        trainer._sparse_thresholds_step = None
        trainer._sparse_threshold_stats = None
        trainer._sparse_step_masks = None
        trainer._sparse_step_masks_step = None
        trainer._sparse_last_logged_step = None

        model = TinyRegressionModel()
        trainer.model = model
        trainer.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        trainer._prepare_inputs = lambda inputs: inputs
        trainer.compute_loss_context_manager = nullcontext
        trainer.compute_loss = lambda model, inputs: F.mse_loss(model(inputs["x"]), inputs["y"])
        trainer.retrieve_c = lambda name: name
        trainer._sparse_prepare_step_state()
        return trainer, model

    def _build_inputs(self):
        x = torch.tensor(
            [
                [0.2, -0.4, 0.1, 0.5],
                [-0.3, 0.7, -0.2, 0.1],
            ],
            dtype=torch.float32,
        )
        y = torch.tensor([[0.25], [-0.15]], dtype=torch.float32)
        return {"x": x, "y": y}

    def _manual_projected_grad(self, trainer, model, inputs, random_vector):
        loss = trainer.compute_loss(model, inputs)
        grads = torch.autograd.grad(
            loss,
            [param for _, param in trainer.named_parameters_to_optim],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        proj = torch.tensor(0.0, dtype=torch.float32)
        for (name, param), grad in zip(trainer.named_parameters_to_optim, grads):
            if grad is None:
                continue
            direction = trainer._zo_effective_perturb_direction(param, random_vector[name])
            proj = proj + torch.sum(grad.detach().float() * direction.detach().float())
        return proj

    def test_seeded_true_directional_derivative_matches_materialized_direction(self):
        trainer, model = self._build_trainer(efficient_zero_order=True)
        inputs = self._build_inputs()
        seed = 12345

        random_vector = trainer._zo_materialize_random_vector(seed)
        manual_proj = self._manual_projected_grad(trainer, model, inputs, random_vector)

        _, td_from_seed = trainer.zo_true_directional_derivative(model, inputs, random_seed=seed)
        _, td_from_vector = trainer.zo_true_directional_derivative(model, inputs, random_vector=random_vector)

        self.assertTrue(torch.allclose(td_from_seed.float(), manual_proj.float(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(td_from_vector.float(), manual_proj.float(), atol=1e-6, rtol=1e-6))

    def test_quzo_true_directional_derivative_uses_u1_for_official_probe(self):
        trainer, model = self._build_trainer(efficient_zero_order=True, zo_quantization_bits=8)
        inputs = self._build_inputs()
        seed = 34567

        random_vector = trainer._zo_materialize_random_vector(seed)
        loss = trainer.compute_loss(model, inputs)
        grads = torch.autograd.grad(
            loss,
            [param for _, param in trainer.named_parameters_to_optim],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        manual_u1 = torch.tensor(0.0, dtype=torch.float32)
        manual_u2 = torch.tensor(0.0, dtype=torch.float32)
        saw_distinct_direction = False
        for (name, _), grad in zip(trainer.named_parameters_to_optim, grads):
            if grad is None:
                continue
            bundle = random_vector[name]
            manual_u1 = manual_u1 + torch.sum(grad.detach().float() * bundle["u1"].detach().float())
            manual_u2 = manual_u2 + torch.sum(grad.detach().float() * bundle["u2"].detach().float())
            saw_distinct_direction = saw_distinct_direction or (not torch.allclose(bundle["u1"], bundle["u2"]))

        _, td_u1 = trainer.zo_true_directional_derivative(
            model,
            inputs,
            random_vector=random_vector,
            probe_direction="u1",
        )
        _, td_u2 = trainer.zo_true_directional_derivative(
            model,
            inputs,
            random_vector=random_vector,
            probe_direction="u2",
        )

        self.assertTrue(saw_distinct_direction)
        self.assertTrue(torch.allclose(td_u1.float(), manual_u1.float(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(td_u2.float(), manual_u2.float(), atol=1e-6, rtol=1e-6))

    def test_efficient_perturbation_reuses_materialized_direction(self):
        trainer, model = self._build_trainer(efficient_zero_order=True)
        seed = 23456
        eps = trainer._get_training_step_size()
        random_vector = trainer._zo_materialize_random_vector(seed)
        originals = {name: param.detach().clone() for name, param in trainer.named_parameters_to_optim}

        trainer.efficient_perturb_parameters(model, seed, random_vector=random_vector)

        for name, param in trainer.named_parameters_to_optim:
            expected = originals[name] + trainer._zo_effective_perturb_direction(param, random_vector[name]) * eps
            self.assertTrue(torch.allclose(param.detach(), expected, atol=1e-7, rtol=1e-6), msg=name)

    def test_non_efficient_random_vector_path_matches_materialized_direction(self):
        trainer, model = self._build_trainer(efficient_zero_order=False)
        inputs = self._build_inputs()

        _, random_vector = trainer.perturb_parameters(model, scaling_factor=0.0)
        manual_proj = self._manual_projected_grad(trainer, model, inputs, random_vector)
        _, td_from_vector = trainer.zo_true_directional_derivative(model, inputs, random_vector=random_vector)

        self.assertTrue(torch.allclose(td_from_vector.float(), manual_proj.float(), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
