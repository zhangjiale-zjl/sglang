"""CPU contracts for ModelSlim W4A8 down-projection TP compensation."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w4a8_int8_moe import (
    ModelSlimW4A8Int8MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestModelSlimW4A8TPCompensation(unittest.TestCase):
    def test_compensation_buffer_uses_layer_tp_size(self):
        for size in (1, 2, 4, 8, 16):
            for prefix in ("w2", "w13"):
                with self.subTest(tp_size=size, prefix=prefix):
                    scheme = ModelSlimW4A8Int8MoE.__new__(ModelSlimW4A8Int8MoE)
                    scheme.weight_prefix = prefix
                    scheme.tp_size = 1
                    scheme.is_per_channel_weight = True
                    layer = torch.nn.Module()
                    layer.moe_tp_size = size
                    scheme.create_weights(
                        layer,
                        num_experts=1,
                        hidden_size=16,
                        intermediate_size_per_partition=16,
                    )
                    width = 16 // size if prefix == "w2" else 1
                    self.assertEqual(
                        getattr(layer, f"{prefix}_scale_bias").shape[-1], width
                    )

    def test_rank_slices_reconstruct_compensation_and_reduced_sum(self):
        full = torch.arange(64, dtype=torch.float32).reshape(4, 16)
        for size in (1, 2, 4, 8, 16):
            with self.subTest(tp_size=size):
                width = 16 // size
                parts = []
                for rank in range(size):
                    part = torch.empty(4, width)
                    FusedMoE._load_per_channel_weight_scale(
                        SimpleNamespace(moe_tp_size=size), part, 2, "w2", full, rank
                    )
                    self.assertTrue(
                        torch.equal(part, full[:, rank * width : (rank + 1) * width])
                    )
                    parts.append(part)
                self.assertTrue(torch.equal(torch.cat(parts, -1), full))
                self.assertTrue(
                    torch.equal(sum(p.sum(-1) for p in parts), full.sum(-1))
                )

    def test_single_column_scale_remains_replicated(self):
        full = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        for rank in range(8):
            with self.subTest(rank=rank):
                target = torch.empty_like(full)
                FusedMoE._load_per_channel_weight_scale(
                    SimpleNamespace(moe_tp_size=8), target, 2, "w2", full, rank
                )
                self.assertTrue(torch.equal(target, full))


if __name__ == "__main__":
    unittest.main()
