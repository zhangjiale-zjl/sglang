"""CPU-only policy tests without importing accelerator-specific cache pools."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def load_policy():
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "UnifiedTreeCore"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_inc_hit_count_and_check"
    )
    # Execute the actual production method, not a second implementation. This
    # policy reads only scalar metadata; its tensor-heavy imports are irrelevant.
    method.args.args[1].annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method.name]


class TestChunkedWriteThrough(unittest.TestCase):
    policy = staticmethod(load_policy())

    def setUp(self):
        self.cache = SimpleNamespace(
            enable_hicache=True,
            enable_external_cache_linker=False,
            is_write_back=False,
            write_through_threshold=1,
        )
        self.node = SimpleNamespace(evicted=False, backuped=False, hit_count=0)

    def test_chunk_is_backed_up_without_counting_reuse(self):
        self.assertTrue(self.policy(self.cache, self.node, chunked=True))
        self.assertEqual(self.node.hit_count, 0)

    def test_chunk_does_not_trigger_selective_write_through(self):
        self.cache.write_through_threshold = 2
        self.assertFalse(self.policy(self.cache, self.node, chunked=True))
        self.assertEqual(self.node.hit_count, 0)

    def test_chunk_does_not_trigger_write_back(self):
        self.cache.is_write_back = True
        self.assertFalse(self.policy(self.cache, self.node, chunked=True))
        self.assertEqual(self.node.hit_count, 0)

    def test_chunk_does_not_trigger_external_linker(self):
        self.cache.enable_external_cache_linker = True
        self.assertFalse(self.policy(self.cache, self.node, chunked=True))

    def test_chunk_does_not_backup_evicted_or_already_backed_up_nodes(self):
        for flag in ("evicted", "backuped"):
            with self.subTest(flag=flag):
                setattr(self.node, flag, True)
                self.assertFalse(self.policy(self.cache, self.node, chunked=True))
                setattr(self.node, flag, False)

    def test_disabled_hicache_does_not_backup_chunks(self):
        self.cache.enable_hicache = False
        self.assertFalse(self.policy(self.cache, self.node, chunked=True))

    def test_independent_reuses_still_reach_selective_threshold(self):
        self.cache.write_through_threshold = 2
        self.assertFalse(self.policy(self.cache, self.node))
        self.assertTrue(self.policy(self.cache, self.node))
        self.assertEqual(self.node.hit_count, 2)


if __name__ == "__main__":
    unittest.main()
