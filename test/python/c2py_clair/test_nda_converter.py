import unittest
import numpy as np

import nda_converter as nc


class TestConstRefArgs(unittest.TestCase):
    """Functions taking arrays as const& or const views."""

    def test_sum_array(self):
        a = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(nc.sum_array(a), 6.0)

    def test_sum_const_view(self):
        a = np.array([1.5, 2.5, 3.5])
        self.assertAlmostEqual(nc.sum_const_view(a), 7.5)


class TestByValue(unittest.TestCase):
    """Functions taking/returning arrays by value."""

    def test_double_array(self):
        a = np.array([1.0, 2.0, 3.0])
        result = nc.double_array(a)
        np.testing.assert_array_almost_equal(result, [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(a, [1.0, 2.0, 3.0])  # original array should be unchanged


class TestMutableViews(unittest.TestCase):
    """Modifying views in-place from Python."""

    def test_fill_matrix(self):
        m = np.zeros((3, 4))
        nc.fill_matrix(m, 7.0)
        np.testing.assert_array_almost_equal(m, np.full((3, 4), 7.0))


class TestReturnArrays(unittest.TestCase):
    """Functions returning arrays by value."""

    def test_make_identity(self):
        np.testing.assert_array_almost_equal(nc.make_identity(3), np.eye(3))


class TestExpressions(unittest.TestCase):
    """Functions returning nda expressions (implicitly converted to arrays)."""

    def test_scale_array(self):
        a = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(nc.scale_array(a, 3.0), [3.0, 6.0, 9.0])

    def test_negate_array(self):
        a = np.array([1.0, -2.0, 3.0])
        np.testing.assert_array_almost_equal(nc.negate_array(a), [-1.0, 2.0, -3.0])

    def test_conj_array(self):
        a = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
        np.testing.assert_array_almost_equal(nc.conj_array(a), np.conj(a))

    def test_add_arrays(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        b = np.array([[5, 6], [7, 8]], dtype=np.int64)
        np.testing.assert_array_equal(nc.add_arrays(a, b), a + b)


class TestArrayContainer(unittest.TestCase):
    """Class holding array with const& getter."""

    def test_set_and_get(self):
        c = nc.ArrayContainer(2, 3)
        v = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c.set_data(v)
        np.testing.assert_array_equal(c.data(), v)

    def test_data_returns_const_ref(self):
        """data_const() returns const& — the returned numpy should be read-only."""
        c = nc.ArrayContainer(2, 2)
        c.set_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
        d = c.data_const()
        self.assertFalse(d.flags['WRITEABLE'])
        with self.assertRaises(ValueError):
            d[0, 0] = 0.0

    def test_data_returns_ref(self):
        """data() returns & — the returned numpy should be writeable."""
        c = nc.ArrayContainer(2, 2)
        c.set_data(np.array([[1.0, 2.0], [3.0, 4.0]]))
        d = c.data()
        self.assertTrue(d.flags['WRITEABLE'])
        d[0, 0] = 10.0
        self.assertEqual(c.data()[0, 0], 10.0)

    def test_data_copy_is_independent(self):
        """data_copy() returns a copy — modifying it does not affect the container."""
        c = nc.ArrayContainer(2, 2)
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        c.set_data(v)
        copy = c.data_copy()
        copy[0, 0] = 999.0
        np.testing.assert_array_almost_equal(c.data(), v)


class TestScalarTypes(unittest.TestCase):
    """Different scalar types."""

    def test_sum_int_array(self):
        a = np.array([1, 2, 3, 4], dtype=np.int64)
        self.assertEqual(nc.sum_int_array(a), 10)

    def test_sum_complex_array(self):
        a = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        self.assertAlmostEqual(nc.sum_complex_array(a), 4 + 6j)


class TestHigherRank(unittest.TestCase):
    """3D array operations."""

    def test_sum_3d(self):
        a = np.ones((2, 3, 4))
        self.assertAlmostEqual(nc.sum_3d(a), 24.0)

    def test_fill_3d(self):
        a = np.zeros((2, 3, 4))
        nc.fill_3d(a, 5.0)
        np.testing.assert_array_almost_equal(a, np.full((2, 3, 4), 5.0))


class TestConversions(unittest.TestCase):
    """Type/rank conversions and enforcement."""

    def test_rank_mismatch_rejected(self):
        with self.assertRaises(TypeError):
            nc.sum_array(np.ones((3, 4)))

    def test_int_to_double_converts(self):
        a = np.array([1, 2, 3], dtype=np.int64)
        self.assertAlmostEqual(nc.sum_array(a), 6.0)

    def test_list_input_converts(self):
        self.assertAlmostEqual(nc.sum_array([1.0, 2.0, 3.0]), 6.0)


if __name__ == "__main__":
    unittest.main()
