"""Host contract tests of actual worker methods; target/GPU is stubbed."""

import ast
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
from suffix_native.cache import SuffixDecodingCache


class Spec:
    width = 33

    @contextmanager
    def override(self, speculative_num_draft_tokens):
        before = self.width
        self.width = speculative_num_draft_tokens
        try:
            yield
        finally:
            self.width = before


spec = Spec()


class TargetStub:
    def _efficient_concat_last_n(self, seq1, seq2, n):
        return (seq1 + seq2)[-n:]

    def forward_batch_generation(self, batch, *args):
        tokens, mask = self._prepare_draft_tokens(batch)
        assert len(tokens) == self.draft_token_num == spec.width
        assert self.draft_tokens_batch[1].shape == tokens.shape
        assert self.tree_mask_batch[1].size == len(tokens) ** 2
        assert np.array_equal(
            mask.reshape(len(tokens), -1), np.tri(len(tokens), dtype=bool)
        )
        return NS(accept_lens=np.asarray([1]), tokens=tokens.copy())


source = ast.parse(
    Path("sglang/python/sglang/srt/speculative/suffix_worker.py").read_text()
)
source.body = [
    n for n in source.body if not isinstance(n, (ast.Import, ast.ImportFrom))
]
namespace = dict(
    np=np,
    NGRAMWorker=TargetStub,
    SuffixDecodingCache=SuffixDecodingCache,
    get_spec=lambda: spec,
    os=__import__("os"),
    json=__import__("json"),
)
exec(compile(source, "suffix_worker.py", "exec"), namespace)
Worker = namespace["SuffixWorker"]


def worker():
    w = Worker.__new__(Worker)
    w.max_width = 33
    w.suffix_cache = SuffixDecodingCache(64, 2)
    w.requests, w.seen = {}, {}
    w.factor, w.offset, w.min_prob, w.fixed = 1.0, 0.0, 0.1, False
    w.trace = None
    w.trace_logits = w.allow_width_probe = False
    for field in ["draft_tokens", "positions"]:
        setattr(w, field, np.empty(33, dtype=np.int64))
    w.tree_mask = np.empty(33**2, dtype=bool)
    for field in ["retrieve_indexes", "retrieve_next_token", "retrieve_next_sibling"]:
        setattr(w, field, np.empty((1, 33), dtype=np.int64))
    for field in [
        "draft_tokens",
        "positions",
        "tree_mask",
        "retrieve_indexes",
        "retrieve_next_token",
        "retrieve_next_sibling",
    ]:
        setattr(w, field + "_batch", [None, None])
    return w


def request(rid="a"):
    return NS(
        rid=rid,
        origin_input_ids=list(range(1, 40)),
        output_ids=[1, 2],
        output_ids_through_stop=[1, 2],
        sampling_params=NS(max_new_tokens=100),
    )


class WorkerContractTests(unittest.TestCase):
    def test_reject_history_penalties_and_nongreedy_prefill(self):
        for name, value in [
            ("frequency_penalty", 0.2),
            ("presence_penalty", 0.2),
            ("repetition_penalty", 1.2),
            ("min_new_tokens", 3),
        ]:
            w, req = worker(), request()
            setattr(req.sampling_params, name, value)
            batch = NS(
                reqs=[req],
                forward_mode=NS(is_decode=lambda: False),
                has_grammar=False,
                return_logprob=False,
                sampling_info=NS(is_all_greedy=True),
            )
            with self.assertRaises(ValueError):
                w.forward_batch_generation(batch)
            self.assertFalse(w.requests)
        batch.sampling_info.is_all_greedy = False
        with self.assertRaises(ValueError):
            worker().forward_batch_generation(batch)

    def test_width_probe_is_explicit_and_bounded(self):
        w, req = worker(), request()
        req.sampling_params.custom_params = {"suffix_probe_width": 8}
        batch = NS(
            reqs=[req], forward_mode=NS(is_decode=lambda: True),
            has_grammar=False, return_logprob=False,
            sampling_info=NS(is_all_greedy=True),
        )
        with self.assertRaises(ValueError):
            w.forward_batch_generation(batch)
        self.assertFalse(w.requests)
        w.allow_width_probe = True
        result = w.forward_batch_generation(batch)
        self.assertEqual(list(result.tokens), [2] + [0] * 7)
        self.assertEqual(w.seen[req.rid], 2)
        req.sampling_params.custom_params["suffix_probe_width"] = 34
        with self.assertRaises(ValueError):
            w.forward_batch_generation(batch)

    def test_true_width_and_committed_cache(self):
        w, req = worker(), request()
        batch = NS(
            reqs=[req],
            forward_mode=NS(is_decode=lambda: True),
            has_grammar=False,
            return_logprob=False,
            sampling_info=NS(is_all_greedy=True),
        )
        result = w.forward_batch_generation(batch)
        self.assertGreater(len(result.tokens), 1)
        self.assertLess(len(result.tokens), w.max_width)
        self.assertEqual(spec.width, 33)
        self.assertEqual(w.seen["a"], 2)  # proposed tail never committed
        w.forward_batch_generation(batch)
        self.assertEqual(w.seen["a"], 2)  # repeated sync does not duplicate
        req.output_ids = [1, 2, 3, 4]
        req.output_ids_through_stop = [1, 2, 3]  # stop-truncated final run
        w.note_request_finished(rid="a", natural_stop=True)
        self.assertFalse(w.requests)
        self.assertFalse(w.suffix_cache.active_requests)
        w.suffix_cache.start_request("b", [90])
        self.assertEqual(w.suffix_cache.speculate("b", [90, 1, 2], 8).token_ids, [3])

    def test_prefill_only_completion(self):
        w, req = worker(), request()
        req.output_ids = req.output_ids_through_stop = []
        w._sync_request(req)
        req.output_ids = req.output_ids_through_stop = [1]
        w.note_request_finished(rid=req.rid, natural_stop=True)
        self.assertFalse(w.requests)
        self.assertFalse(w.suffix_cache.active_requests)
        self.assertIn(req.rid, w.suffix_cache.cached_requests)
        self.assertEqual(w.suffix_cache._global_tree.check_integrity(), "")

    def test_abort_and_flush_remove_state(self):
        w, req = worker(), request()
        w._sync_request(req)
        w.discard_request(req.rid)
        self.assertFalse(w.requests)
        self.assertFalse(w.suffix_cache.cached_requests)
        w._sync_request(req)  # request ID reuse after abort
        w.clear_cache_pool()
        self.assertFalse(w.requests)
        self.assertFalse(w.suffix_cache.cached_requests)
        self.assertFalse(w.suffix_cache.active_requests)

    def test_empty_match_one_row_and_output_cap(self):
        w, req = worker(), request()
        req.origin_input_ids = [91]
        req.output_ids = req.output_ids_through_stop = [92]
        batch = NS(
            reqs=[req],
            forward_mode=NS(is_decode=lambda: True),
            has_grammar=False,
            return_logprob=False,
            sampling_info=NS(is_all_greedy=True),
        )
        self.assertEqual(len(w.forward_batch_generation(batch).tokens), 1)
        req.sampling_params.max_new_tokens = 2
        self.assertEqual(len(w.forward_batch_generation(batch).tokens), 1)


if __name__ == "__main__":
    unittest.main()
