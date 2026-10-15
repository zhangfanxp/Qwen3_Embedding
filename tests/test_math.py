"""不下载模型即可运行的核心数学逻辑单元测试。"""
import unittest

import numpy as np

from embedding_engine import cosine_pairs


class CosineTest(unittest.TestCase):
    def test_three_vectors(self):
        names = ["A", "B", "C"]
        # A 与 B 同向，A 与 C 正交
        vectors = np.array([[1, 0], [2, 0], [0, 1]], dtype=np.float32)
        pairs = cosine_pairs(names, vectors)
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(pairs[0]["cosine_similarity"], 1.0)
        self.assertAlmostEqual(pairs[1]["cosine_similarity"], 0.0)
        self.assertAlmostEqual(pairs[2]["cosine_similarity"], 0.0)

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            cosine_pairs(["A", "B"], np.array([[0, 0], [1, 0]]))


if __name__ == "__main__":
    unittest.main()
