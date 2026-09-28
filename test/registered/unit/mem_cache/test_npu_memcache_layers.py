"""MemCache 分层 buffer 与单缓冲零拷贝接口的分发测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store import (
    NpuMemcacheStore,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuMemcacheLayers(CustomTestCase):
    def setUp(self):
        # 绕开真实 MMC 初始化，只验证 key、buffer 形状和返回码语义。
        self.backend = NpuMemcacheStore.__new__(NpuMemcacheStore)
        self.backend.store = Mock()
        self.backend.is_mla_backend = True
        self.backend.mla_suffix = ""
        self.backend.extra_backend_tag = None
        self.backend.enable_storage_metrics = False
        self.backend.mem_pool_host = Mock()
        self.backend.mem_pool_host.page_size = 1
        self.backend.mem_pool_host.layout = "layer_first"
        self.backend.mem_pool_host.get_page_buffer_meta.return_value = (
            [11, 12, 21, 22],
            [4, 4, 4, 4],
        )
        self.backend.registered_pools = {}
        self.backend.storage_config = None
        self.indices = torch.tensor([0, 1])

    def test_v1_get_uses_layers_and_converts_result_codes(self):
        self.backend.store.batch_get_into_layers.return_value = [0, 5]

        result = self.backend.batch_get_v1(["a", "b"], self.indices)

        self.assertEqual(result, [True, False])
        self.backend.store.batch_get_into_layers.assert_called_once_with(
            ["a__k", "b__k"], [[11, 12], [21, 22]], [[4, 4], [4, 4]]
        )
        self.backend.store.batch_get_into.assert_not_called()

    def test_v1_set_skips_existing_key_and_uses_layers(self):
        self.backend.store.batch_is_exist.return_value = [1, 0]
        self.backend.store.batch_put_from_layers.return_value = [0]

        result = self.backend.batch_set_v1(["a", "b"], self.indices)

        self.assertEqual(result, [True, True])
        self.backend.store.batch_put_from_layers.assert_called_once_with(
            ["b__k"], [[21, 22]], [[4, 4]]
        )
        self.backend.store.batch_put_from.assert_not_called()

    def test_v2_get_and_set_use_layers(self):
        pool = Mock()
        pool.page_size = 1
        pool.get_page_buffer_meta.return_value = ([11, 12, 21, 22], [4] * 4)
        self.backend.registered_pools[PoolName.INDEXER] = pool
        transfer = PoolTransfer(
            name=PoolName.INDEXER, keys=["a", "b"], host_indices=self.indices
        )
        self.backend.store.batch_get_into_layers.return_value = [0, 0]

        self.assertEqual(
            self.backend.batch_get_v2([transfer]), {PoolName.INDEXER: [True, True]}
        )
        self.backend.store.batch_get_into_layers.assert_called_once_with(
            ["a__indexer", "b__indexer"],
            [[11, 12], [21, 22]],
            [[4, 4], [4, 4]],
        )

        self.backend.store.batch_is_exist.return_value = [1, 0]
        self.backend.store.batch_put_from_layers.return_value = [0]
        self.assertEqual(
            self.backend.batch_set_v2([transfer]), {PoolName.INDEXER: [True, True]}
        )
        self.backend.store.batch_put_from_layers.assert_called_once_with(
            ["b__indexer"], [[21, 22]], [[4, 4]]
        )

    def test_flat_buffer_keeps_original_interface(self):
        self.backend.store.batch_get_into.return_value = [0, 3]
        result = self.backend._get_batch_zero_copy_impl(
            ["a", "b"], [11, 21], [4, 4]
        )
        self.assertEqual(result, [4, -3])
        self.backend.store.batch_get_into_layers.assert_not_called()

        self.backend.store.batch_put_from.return_value = [0]
        self.assertEqual(
            self.backend._put_batch_zero_copy_impl(["a"], [11], [4]), [0]
        )
        self.backend.store.batch_put_from_layers.assert_not_called()


if __name__ == "__main__":
    unittest.main()
