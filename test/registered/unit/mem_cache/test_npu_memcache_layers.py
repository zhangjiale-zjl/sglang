"""MemCache 分层 buffer 与单缓冲零拷贝接口的分发测试。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
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


class TestNpuMemcacheDeepSeekV4(CustomTestCase):
    def setUp(self):
        self.backend = NpuMemcacheStore.__new__(NpuMemcacheStore)
        self.backend.store = Mock()
        self.backend.is_mla_backend = True
        self.backend.mla_suffix = ""
        self.backend.mha_suffix = "0"
        self.backend.extra_backend_tag = None
        self.backend.enable_storage_metrics = False
        self.backend.registered_pools = {}
        self.backend.mem_pool_host = SimpleNamespace(kv_buffer=None, page_size=1)
        self.indices = torch.tensor([0, 1])

    def test_logical_anchor_has_no_buffer_or_standalone_storage_hit(self):
        self.backend.register_mem_pool_host(self.backend.mem_pool_host)

        self.assertEqual(self.backend.gb_per_page, 0)
        self.backend.store.register_buffer.assert_not_called()
        self.assertEqual(
            self.backend.batch_get_v1(["a", "b"], self.indices), [True, True]
        )
        self.assertEqual(
            self.backend.batch_set_v1(["a", "b"], self.indices), [True, True]
        )
        self.assertEqual(self.backend.batch_exists(["a", "b"]), 0)
        self.assertEqual(self.backend.batch_exists_v2(["a", "b"]).kv_hit_pages, 0)
        self.backend.store.batch_is_exist.assert_not_called()

    def test_v4_side_pool_registers_each_layer_buffer(self):
        tensors = [torch.empty(4), torch.empty(4)]
        pool = Mock()
        pool.get_hybrid_pool_buffer.return_value = tensors
        self.backend.register_buffer = Mock()

        self.backend.register_mem_host_pool_v2(pool, PoolName.DEEPSEEK_V4_C4)

        self.assertIs(self.backend.registered_pools[PoolName.DEEPSEEK_V4_C4], pool)
        self.assertEqual(self.backend.register_buffer.call_count, 2)

    def test_v4_component_key_suffixes(self):
        for name in (
            PoolName.DEEPSEEK_V4_C1,
            PoolName.DEEPSEEK_V4_C1_INDEXER,
            PoolName.DEEPSEEK_V4_C2,
            PoolName.DEEPSEEK_V4_C4_ROPE,
            PoolName.DEEPSEEK_V4_C128,
            PoolName.DEEPSEEK_V4_C128_ROPE,
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        ):
            with self.subTest(pool=name):
                self.backend.registered_pools[name] = Mock()
                keys, multiplier = self.backend._get_hybrid_page_component_keys(
                    ["page"], PoolTransfer(name=name)
                )
                self.assertEqual(keys, [f"page__{name}"])
                self.assertEqual(multiplier, 1)

    def test_v4_keys_and_layered_get_set(self):
        pool = Mock()
        pool.page_size = 1
        pool.get_page_buffer_meta.return_value = ([11, 12, 21, 22], [4] * 4)
        self.backend.registered_pools[PoolName.DEEPSEEK_V4_C4] = pool
        transfer = PoolTransfer(
            name=PoolName.DEEPSEEK_V4_C4,
            keys=["a", "b"],
            host_indices=self.indices,
        )
        self.backend.store.batch_get_into_layers.return_value = [0, 0]

        self.assertEqual(
            self.backend.batch_get_v2([transfer]),
            {PoolName.DEEPSEEK_V4_C4: [True, True]},
        )
        self.backend.store.batch_get_into_layers.assert_called_once_with(
            ["a__deepseek_v4_c4", "b__deepseek_v4_c4"],
            [[11, 12], [21, 22]],
            [[4, 4], [4, 4]],
        )

        self.backend.store.batch_is_exist.return_value = [1, 0]
        self.backend.store.batch_put_from_layers.return_value = [0]
        self.assertEqual(
            self.backend.batch_set_v2([transfer]),
            {PoolName.DEEPSEEK_V4_C4: [True, True]},
        )
        self.backend.store.batch_put_from_layers.assert_called_once_with(
            ["b__deepseek_v4_c4"], [[21, 22]], [[4, 4]]
        )

    def test_v4_hit_requires_common_restorable_prefix(self):
        self.backend.registered_pools[PoolName.DEEPSEEK_V4_C4] = Mock()
        self.backend.registered_pools[PoolName.DEEPSEEK_V4_C4_STATE] = Mock()
        self.backend.store.batch_is_exist.side_effect = [
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ]
        transfers = [
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
            PoolTransfer(
                name=PoolName.DEEPSEEK_V4_C4_STATE,
                keys=["b", "c"],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

        result = self.backend.batch_exists_v2(["a", "b", "c", "d"], transfers)

        self.assertEqual(result.kv_hit_pages, 3)
        self.assertEqual(result.restorable_prefix_pages, [3])
        self.assertEqual(result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C4], 4)
        self.assertEqual(result.extra_pool_hit_pages[PoolName.DEEPSEEK_V4_C4_STATE], 3)


if __name__ == "__main__":
    unittest.main()
