import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "large_models"))

from trainer import OurTrainer  # noqa: E402
from quzo import quantize_tensor  # noqa: E402


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


class LargeQuzoProbeDirectionTest(unittest.TestCase):
    def _build_trainer(self):
        torch.manual_seed(0)
        trainer = OurTrainer.__new__(OurTrainer)
        trainer.args = SimpleNamespace(
            non_diff=False,
            n_gpu=1,
            seed=11,
            zo_quantization_bits=8,
            sparse_ratio=1.0,
            sparse_log_active_fraction=False,
            logging_steps=10,
            weight_decay=0.0,
            load_float16=True,
            load_bfloat16=False,
            load_int8=False,
            zo_eps=1e-3,
        )
        trainer.state = SimpleNamespace(global_step=0)
        trainer._sparse_step_masks = None
        trainer._sparse_step_masks_step = None
        trainer._sparse_last_logged_step = None
        trainer._prepare_inputs = lambda inputs: inputs
        trainer.compute_loss_context_manager = nullcontext
        trainer.compute_loss = lambda model, inputs: F.mse_loss(model(inputs["x"]), inputs["y"])
        model = TinyRegressionModel()
        trainer.model = model
        trainer.named_parameters_to_optim = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
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

    def test_quzo_official_probe_uses_u1_and_perturbation_matches_bundle(self):
        trainer, model = self._build_trainer()
        inputs = self._build_inputs()
        seed = 45678
        eps = float(trainer.args.zo_eps)

        originals = {name: param.detach().clone() for name, param in trainer.named_parameters_to_optim}
        bundles = {name: trainer._quzo_get_bundle(name, param, seed) for name, param in trainer.named_parameters_to_optim}

        trainer.zo_perturb_parameters(random_seed=seed, scaling_factor=1)
        for name, param in trainer.named_parameters_to_optim:
            bundle = bundles[name]
            expected = quantize_tensor(
                originals[name].detach().float() + bundle["u1"].detach().float() * eps,
                trainer._zo_quant_bits(),
                seed=int(bundle["state_seed"].item()),
                target_dtype=originals[name].dtype,
            )
            self.assertTrue(torch.allclose(param.detach(), expected, atol=1e-7, rtol=1e-6), msg=name)

        trainer.zo_perturb_parameters(random_seed=seed, scaling_factor=-1)

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
            bundle = bundles[name]
            manual_u1 = manual_u1 + torch.sum(grad.detach().float() * bundle["u1"].detach().float())
            manual_u2 = manual_u2 + torch.sum(grad.detach().float() * bundle["u2"].detach().float())
            saw_distinct_direction = saw_distinct_direction or (not torch.allclose(bundle["u1"], bundle["u2"]))

        _, td_u1 = trainer.zo_true_directional_derivative(model, inputs, random_seed=seed, probe_direction="u1")
        _, td_u2 = trainer.zo_true_directional_derivative(model, inputs, random_seed=seed, probe_direction="u2")

        self.assertTrue(saw_distinct_direction)
        self.assertTrue(torch.allclose(td_u1.float(), manual_u1.float(), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(td_u2.float(), manual_u2.float(), atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
