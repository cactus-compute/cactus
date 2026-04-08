import unittest
import numpy as np
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.graph import Graph, Tensor


class TestGraphElementwise(unittest.TestCase):

    def test_pow_abs(self):
        g = Graph()
        a = g.input((4,))
        y = a.pow(2.0).abs()

        g.set_input(a, np.array([-2.0, -1.0, 2.0, 3.0], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([4.0, 1.0, 4.0, 9.0], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_subtract(self):
        g = Graph()
        a = g.input((4,))
        b = g.input((4,))
        y = a - b

        g.set_input(a, np.array([5, 6, 7, 8], dtype=np.float16))
        g.set_input(b, np.array([1, 2, 3, 4], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([4, 4, 4, 4], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_add(self):
        g = Graph()
        a = g.input((4,))
        b = g.input((4,))
        y = a + b

        g.set_input(a, np.array([1, 2, 3, 4], dtype=np.float16))
        g.set_input(b, np.array([10, 20, 30, 40], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([11, 22, 33, 44], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_multiply(self):
        g = Graph()
        a = g.input((4,))
        b = g.input((4,))
        y = a * b

        g.set_input(a, np.array([2, 3, 4, 5], dtype=np.float16))
        g.set_input(b, np.array([3, 4, 5, 6], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([6, 12, 20, 30], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_divide(self):
        g = Graph()
        a = g.input((4,))
        b = g.input((4,))
        y = a / b

        g.set_input(a, np.array([10, 20, 30, 40], dtype=np.float16))
        g.set_input(b, np.array([2, 4, 5, 8], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([5, 5, 6, 5], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)


class TestGraphComposed(unittest.TestCase):

    def test_composed_inference_graph(self):
        g = Graph()

        a = g.input((2, 2))
        b = g.input((2, 2))

        diff = a - b
        summ = a + b
        mixed = (diff * summ) / b
        feat = mixed.abs().pow(2.0).view((4,))
        out_node = g.cat([feat, feat], axis=0)

        a_data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float16)
        b_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        g.set_input(a, a_data)
        g.set_input(b, b_data)
        g.execute()

        out = out_node.numpy()
        base = ((a_data - b_data) * (a_data + b_data)) / b_data
        expected = np.concatenate(
            [np.abs(base).reshape(4) ** 2, np.abs(base).reshape(4) ** 2]
        ).astype(np.float16)

        self.assertEqual(out.shape, (8,))
        np.testing.assert_allclose(out, expected, atol=1e-2)


class TestGraphTensorOps(unittest.TestCase):

    def test_view(self):
        g = Graph()
        a = g.input((2, 3))
        y = a.view((6,))

        g.set_input(a, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_flatten(self):
        g = Graph()
        a = g.input((2, 3))
        y = a.flatten()

        g.set_input(a, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([1, 2, 3, 4, 5, 6], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_concat(self):
        g = Graph()
        a = g.input((2, 2))
        b = g.input((2, 2))
        y = a.concat(b, axis=0)

        g.set_input(a, np.array([[1, 2], [3, 4]], dtype=np.float16))
        g.set_input(b, np.array([[5, 6], [7, 8]], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_cat_1d(self):
        g = Graph()
        a = g.input((3,))
        b = g.input((2,))
        y = g.cat([a, b], axis=0)

        g.set_input(a, np.array([1, 2, 3], dtype=np.float16))
        g.set_input(b, np.array([4, 5], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([1, 2, 3, 4, 5], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_cat_2d_axis1(self):
        g = Graph()
        a = g.input((2, 2))
        b = g.input((2, 3))
        y = g.cat([a, b], axis=1)

        g.set_input(a, np.array([[1, 2], [3, 4]], dtype=np.float16))
        g.set_input(b, np.array([[5, 6, 7], [8, 9, 10]], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)


class TestGraphActivations(unittest.TestCase):

    def test_relu(self):
        g = Graph()
        a = g.input((4,))
        y = a.relu()

        g.set_input(a, np.array([-2, -1, 0, 3], dtype=np.float16))
        g.execute()

        out = y.numpy()
        expected = np.array([0, 0, 0, 3], dtype=np.float16)
        np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_sigmoid(self):
        g = Graph()
        a = g.input((3,))
        y = a.sigmoid()

        data = np.array([0, 2, -2], dtype=np.float16)
        g.set_input(a, data)
        g.execute()

        out = y.numpy()
        expected = (1.0 / (1.0 + np.exp(-data.astype(np.float32)))).astype(np.float16)
        np.testing.assert_allclose(out, expected, atol=5e-2)

    def test_tanh(self):
        g = Graph()
        a = g.input((3,))
        y = a.tanh()

        data = np.array([0, 1, -1], dtype=np.float16)
        g.set_input(a, data)
        g.execute()

        out = y.numpy()
        expected = np.tanh(data.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(out, expected, atol=5e-2)


class TestGraphSoftmax(unittest.TestCase):

    def test_softmax_2d(self):
        g = Graph()
        a = g.input((2, 4))
        y = a.softmax()

        data = np.array([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=np.float16)
        g.set_input(a, data)
        g.execute()

        out = y.numpy()
        # softmax over last axis
        f = data.astype(np.float32)
        e = np.exp(f - f.max(axis=-1, keepdims=True))
        expected = (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)
        np.testing.assert_allclose(out, expected, atol=5e-2)


class TestGraphSaveLoad(unittest.TestCase):

    def _rebind_tensor(self, graph, tensor):
        return Tensor(graph, tensor.id, tensor.shape, tensor.dtype)

    def test_save_load_roundtrip_composed_graph(self):
        g = Graph()

        a = g.input((2, 2))
        b = g.input((2, 2))
        diff = a - b
        summ = a + b
        mixed = (diff * summ) / b
        feat = mixed.abs().pow(2.0).view((4,))
        out_node = g.cat([feat, feat], axis=0)

        a_data = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float16)
        b_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)

        g.set_input(a, a_data)
        g.set_input(b, b_data)
        g.execute()
        expected = out_node.numpy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip_graph.cg"
            g.save(path)

            loaded = Graph.load(path)
            loaded_a = self._rebind_tensor(loaded, a)
            loaded_b = self._rebind_tensor(loaded, b)
            loaded_out = self._rebind_tensor(loaded, out_node)

            loaded.set_input(loaded_a, a_data)
            loaded.set_input(loaded_b, b_data)
            loaded.execute()

            out = loaded_out.numpy()
            self.assertEqual(out.shape, expected.shape)
            np.testing.assert_allclose(out, expected, atol=1e-2)

    def test_save_load_roundtrip_softmax_cat(self):
        g = Graph()

        a = g.input((2, 3))
        b = g.input((2, 3))
        joined = g.cat([a, b], axis=1)
        out_node = joined.softmax(axis=1)

        a_data = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=np.float16)
        b_data = np.array([[4.0, 5.0, 6.0], [3.5, 4.5, 5.5]], dtype=np.float16)

        g.set_input(a, a_data)
        g.set_input(b, b_data)
        g.execute()
        expected = out_node.numpy()
        expected_info = g.output_info(out_node)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "softmax_cat_graph.cg"
            g.save(path)

            loaded = Graph.load(path)
            loaded_a = self._rebind_tensor(loaded, a)
            loaded_b = self._rebind_tensor(loaded, b)
            loaded_out = self._rebind_tensor(loaded, out_node)

            loaded.set_input(loaded_a, a_data)
            loaded.set_input(loaded_b, b_data)
            loaded.execute()

            out = loaded_out.numpy()
            info = loaded.output_info(loaded_out)

            self.assertEqual(info["shape"], expected_info["shape"])
            self.assertEqual(info["precision"], expected_info["precision"])
            np.testing.assert_allclose(out, expected, atol=5e-2)


if __name__ == "__main__":
    unittest.main()
