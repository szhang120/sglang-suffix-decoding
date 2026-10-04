import random
import unittest

from suffix_native.cache import SuffixDecodingCache


class CacheTests(unittest.TestCase):
    def test_prompt_and_global_are_separate(self):
        c = SuffixDecodingCache(64, 2)
        c.start_request("a", [8, 9, 10, 11, 8, 9])
        self.assertEqual(c.speculate("a", [0, 8, 9], 8).token_ids[:2], [10, 11])
        c.add_active_response("a", [20, 21, 22, 23])
        c.stop_request("a")
        c.start_request("b", [99])
        self.assertEqual(c.speculate("b", [0, 20, 21], 8).token_ids[:2], [22, 23])
        self.assertEqual(c.speculate("b", [0, 8, 9], 8).token_ids, [])

    def test_eviction_and_id_reuse(self):
        c = SuffixDecodingCache(16, 1)
        for rid, tokens in [("a", [1, 2, 3]), ("b", [4, 5, 6])]:
            c.start_request(rid, [])
            c.add_active_response(rid, tokens)
            c.stop_request(rid)
        self.assertEqual(list(c.cached_requests), ["b"])
        c.start_request("b", [7, 8])
        self.assertEqual(c.speculate("b", [0, 4, 5], 4).token_ids, [])
        c.stop_request("b")
        self.assertFalse(c.active_requests)

    def test_adaptive_length_and_integrity(self):
        c = SuffixDecodingCache(64, 4)
        c.start_request("a", list(range(1, 40)))
        for match in range(1, 20):
            draft = c.speculate(
                "a",
                [999] + list(range(1, match + 1)),
                32,
                max_spec_factor=1,
                min_token_prob=0.1,
            )
            self.assertLessEqual(len(draft.token_ids), draft.match_len)
            self.assertEqual(
                draft.token_ids,
                list(range(match + 1, match + 1 + len(draft.token_ids))),
            )
            self.assertEqual(draft.parents, list(range(-1, len(draft.token_ids) - 1)))
        self.assertEqual(c._global_tree.check_integrity(), "")
        self.assertEqual(c._local_trees["a"].check_integrity(), "")
        self.assertIsNotNone(c.speculate("a", [0, 1, 2]))  # default depth bug fix

    def test_oracle_greedy_simulation(self):
        # Algorithmic oracle only: no model/KV/GPU claims. Arbitrary wrong
        # drafts must still yield exactly the target sequence with a bonus.
        rng = random.Random(42)
        c = SuffixDecodingCache(32, 5)
        for rid in range(100):
            prompt = [rng.randrange(8) for _ in range(30)]
            target = [rng.randrange(8) for _ in range(80)]
            c.start_request(rid, prompt)
            out = target[:1]
            c.add_active_response(rid, out)
            while len(out) < len(target):
                draft = c.speculate(
                    rid, prompt + out, min(16, len(target) - len(out) - 1)
                )
                n = 0
                while (
                    n < len(draft.token_ids)
                    and draft.token_ids[n] == target[len(out) + n]
                ):
                    n += 1
                committed = (
                    draft.token_ids[:n] + target[len(out) + n : len(out) + n + 1]
                )
                out.extend(committed)
                c.add_active_response(rid, committed)
            self.assertEqual(out, target)
            self.assertEqual(c._global_tree.check_integrity(), "")
            self.assertEqual(c._local_trees[rid].check_integrity(), "")
            c.stop_request(rid)


if __name__ == "__main__":
    unittest.main()
