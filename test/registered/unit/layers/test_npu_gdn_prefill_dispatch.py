import unittest
from unittest.mock import Mock

import torch

from sglang.srt.hardware_backend.npu.attention.gdn_prefill import run_gdn_prefill
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNpuGdnPrefillDispatch(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(4, 2, 3)
        self.k = torch.randn(4, 2, 3)
        self.v = torch.randn(4, 2, 3)
        self.g = torch.randn(4, 2)
        self.beta = torch.randn(4, 2)
        self.states = torch.randn(3, 2, 3, 3)
        self.indices = torch.tensor([2])
        self.lengths = torch.tensor([0, 4])
        self.result = (Mock(), Mock(), Mock())

    def run_prefill(self, native, dispatcher):
        return run_gdn_prefill(
            native,
            dispatcher,
            self.q,
            self.k,
            self.v,
            self.g,
            self.beta,
            self.states,
            self.indices,
            self.lengths,
        )

    def test_native_operator_receives_selected_initial_state(self):
        native = Mock(return_value=self.result)
        dispatcher = Mock()
        self.assertIs(self.run_prefill(native, dispatcher), self.result)
        dispatcher.extend.assert_not_called()
        args = native.call_args.args
        self.assertIs(args[0], self.q)
        self.assertTrue(torch.equal(args[5], self.states[self.indices]))
        self.assertIs(args[6], self.lengths)

    def test_fallback_preserves_state_pool_indices_and_checkpoints(self):
        dispatcher = Mock()
        dispatcher.extend.return_value = self.result
        self.assertIs(self.run_prefill(None, dispatcher), self.result)
        args = dispatcher.extend.call_args.kwargs
        for name in ("q", "k", "v"):
            self.assertEqual(args[name].shape, (1, 4, 2, 3))
            self.assertTrue(torch.equal(args[name][0], getattr(self, name)))
        self.assertIs(args["ssm_states"], self.states)
        self.assertIs(args["cache_indices"], self.indices)
        self.assertIs(args["query_start_loc"], self.lengths)

    def test_native_failure_is_not_silently_retried_with_another_kernel(self):
        native = Mock(side_effect=RuntimeError("native failed"))
        dispatcher = Mock()
        with self.assertRaisesRegex(RuntimeError, "native failed"):
            self.run_prefill(native, dispatcher)
        dispatcher.extend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
