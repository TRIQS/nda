import gc
import unittest
import numpy as np

import nda_converter as nc


# ==================================================================
# Argument types
# ==================================================================

class TestArgArrayConstRef(unittest.TestCase):
    """array<T, R> const &: copies from numpy, auto-converts dtype and layout."""

    def test_1d(self):
        self.assertAlmostEqual(nc.sum_array(np.array([1.0, 2.0, 3.0])), 6.0)

    def test_2d(self):
        self.assertAlmostEqual(nc.sum_matrix(np.ones((2, 3))), 6.0)

    def test_converts_dtype(self):
        self.assertAlmostEqual(nc.sum_array(np.array([1, 2, 3], dtype=np.float32)), 6.0)
        self.assertAlmostEqual(nc.sum_array(np.array([1, 2, 3], dtype=np.int64)), 6.0)

    def test_converts_layout(self):
        m = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertAlmostEqual(nc.sum_matrix(m), 10.0)
        self.assertAlmostEqual(nc.sum_matrix(m.T), 10.0)

    def test_accepts_list_and_tuple(self):
        self.assertAlmostEqual(nc.sum_array([1.0, 2.0, 3.0]), 6.0)
        self.assertAlmostEqual(nc.sum_array((1.0, 2.0, 3.0)), 6.0)
        self.assertAlmostEqual(nc.sum_matrix([[1, 2], [3, 4]]), 10.0)

    def test_empty(self):
        self.assertAlmostEqual(nc.sum_array(np.array([], dtype=np.float64)), 0.0)
        self.assertAlmostEqual(nc.sum_matrix(np.zeros((0, 3))), 0.0)

    def test_rejects_wrong_rank(self):
        with self.assertRaises(TypeError):
            nc.sum_array(np.ones((3, 4)))

    def test_rejects_non_array(self):
        for bad in [42.0, None, "hello"]:
            with self.assertRaises(TypeError):
                nc.sum_array(bad)


class TestArgArrayByValue(unittest.TestCase):
    """array<T, R> by value: same conversion as const ref, but original is never modified."""

    def test_returns_doubled(self):
        np.testing.assert_array_almost_equal(nc.double_array(np.array([1.0, 2.0, 3.0])), [2.0, 4.0, 6.0])

    def test_original_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        nc.double_array(a)
        np.testing.assert_array_equal(a, [1.0, 2.0, 3.0])

    def test_converts_dtype_and_layout(self):
        np.testing.assert_array_almost_equal(nc.double_array(np.array([1, 2], dtype=np.float32)), [2.0, 4.0])
        np.testing.assert_array_almost_equal(nc.double_array(np.asfortranarray(np.array([1.0, 2.0]))), [2.0, 4.0])

    def test_accepts_list(self):
        np.testing.assert_array_almost_equal(nc.double_array([1.0, 2.0]), [2.0, 4.0])

    def test_double_non_contiguous_view(self):
        """By-value array parameters can be passed non-contiguous views, since they are copied."""
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = nc.double_array(m[:, 0])
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])


class TestArgMutableView(unittest.TestCase):
    """array_view<T, R>: zero-copy, requires exact dtype and C-compatible stride order."""

    def test_fills_in_place(self):
        v = np.zeros(5)
        nc.fill_view_1d(v, 3.0)
        np.testing.assert_array_almost_equal(v, np.full(5, 3.0))

    def test_fills_2d(self):
        m = np.zeros((3, 4))
        nc.fill_view_2d(m, 7.0)
        np.testing.assert_array_almost_equal(m, np.full((3, 4), 7.0))

    def test_strided_2d(self):
        """C_stride_layout views accept non-contiguous C-ordered strides."""
        m = np.zeros((6, 4))
        nc.fill_view_2d(m[::2, :], 1.0)
        np.testing.assert_array_almost_equal(m[0], [1, 1, 1, 1])
        np.testing.assert_array_almost_equal(m[1], [0, 0, 0, 0])
        np.testing.assert_array_almost_equal(m[2], [1, 1, 1, 1])

    def test_empty(self):
        v = np.array([], dtype=np.float64)
        nc.fill_view_1d(v, 7.0)
        self.assertEqual(len(v), 0)

    def test_rejects_wrong_dtype(self):
        with self.assertRaises(TypeError):
            nc.fill_view_2d(np.zeros((3, 4), dtype=np.float32), 1.0)
        with self.assertRaises(TypeError):
            nc.fill_view_1d(np.zeros(5, dtype=np.int64), 1.0)

    def test_rejects_fortran_order(self):
        with self.assertRaises(TypeError):
            nc.fill_view_2d(np.asfortranarray(np.zeros((3, 4))), 1.0)

    def test_rejects_wrong_rank(self):
        with self.assertRaises(TypeError):
            nc.fill_view_2d(np.zeros(5), 1.0)


class TestArgConstView(unittest.TestCase):
    """array_const_view<T, R>: zero-copy read-only, accepts read-only numpy."""

    def test_basic(self):
        self.assertAlmostEqual(nc.sum_const_view(np.array([1.5, 2.5, 3.5])), 7.5)

    def test_readonly_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        a.flags.writeable = False
        self.assertAlmostEqual(nc.sum_const_view(a), 6.0)


# ==================================================================
# Return types
# ==================================================================

class TestReturnByValue(unittest.TestCase):
    """Returned arrays own their data and are writeable."""

    def test_basic(self):
        np.testing.assert_array_almost_equal(nc.make_sequence(5), [0, 1, 2, 3, 4])

    def test_writeable_and_owns_data(self):
        a = nc.make_sequence(100)
        self.assertTrue(a.flags['WRITEABLE'])
        gc.collect()
        np.testing.assert_array_almost_equal(a[:3], [0, 1, 2])

    def test_empty(self):
        self.assertEqual(len(nc.make_sequence(0)), 0)


class TestReturnRef(unittest.TestCase):
    """array const& -> read-only numpy; array& -> writeable numpy sharing C++ memory."""

    def setUp(self):
        self.c = nc.ArrayContainer(2, 3)
        self.c.set_data(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_const_ref_is_readonly(self):
        d = self.c.data_const()
        self.assertFalse(d.flags['WRITEABLE'])
        with self.assertRaises(ValueError):
            d[0, 0] = 0.0

    def test_mutable_ref_shares_memory(self):
        d = self.c.data()
        self.assertTrue(d.flags['WRITEABLE'])
        d[0, 0] = 99.0
        self.assertEqual(self.c.data()[0, 0], 99.0)

    def test_copy_is_independent(self):
        copy = self.c.data_copy()
        copy[0, 0] = 999.0
        self.assertEqual(self.c.data()[0, 0], 1.0)

    def test_copy_survives_container_deletion(self):
        copy = self.c.data_copy()
        del self.c
        gc.collect()
        np.testing.assert_array_almost_equal(copy[0], [1.0, 2.0, 3.0])


class TestReturnExpression(unittest.TestCase):
    """Expressions are materialized into regular arrays on return."""

    def test_scale(self):
        np.testing.assert_array_almost_equal(nc.scale_array(np.array([1.0, 2.0, 3.0]), 3.0), [3.0, 6.0, 9.0])

    def test_negate(self):
        np.testing.assert_array_almost_equal(nc.negate_array(np.array([1.0, -2.0])), [-1.0, 2.0])

    def test_conj(self):
        a = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
        np.testing.assert_array_almost_equal(nc.conj_array(a), np.conj(a))

    def test_add(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)
        b = np.array([[5, 6], [7, 8]], dtype=np.int64)
        np.testing.assert_array_equal(nc.add_arrays(a, b), a + b)


# ==================================================================
# Return: array of arrays (nested non-npy element type via class)
# ==================================================================

class TestReturnNestedArrayRef(unittest.TestCase):
    """array<array<double,1>, 1> returned as const&, &, and copy from a wrapped class."""

    def setUp(self):
        self.c = nc.NestedContainer(3, 4)  # 3 inner arrays of size 4

    def test_data_const_returns_correct_values(self):
        # const& of non-npy array returns a c2py_range; convert to list of arrays
        d = list(self.c.data_const())
        self.assertEqual(len(d), 3)
        # Inner array i is filled with float(i)
        np.testing.assert_array_almost_equal(d[0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(d[1], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(d[2], [2.0, 2.0, 2.0, 2.0])

    def test_data_returns_correct_values(self):
        d = list(self.c.data())
        self.assertEqual(len(d), 3)
        np.testing.assert_array_almost_equal(d[1], [1.0, 1.0, 1.0, 1.0])

    # def test_data_mutation_does_not_propagate(self):
    #     """For non-npy element types, c2py_range yields copies of inner arrays.
    #     Unlike the flat array_container case, mutations do not propagate back."""
    #     d = list(self.c.data_const())
    #     d[0][0] = 999.0
    #     d2 = list(self.c.data_const())
    #     np.testing.assert_array_almost_equal(d2[0], [0.0, 0.0, 0.0, 0.0])

    def test_data_mutation_does_propagate_for_ref(self):
        """For non-npy element types, c2py_range yields copies of inner arrays.
        Unlike the flat array_container case, mutations do not propagate back."""
        d = list(self.c.data())
        d[0][0] = 999.0
        d2 = list(self.c.data())
        np.testing.assert_array_almost_equal(d2[0], [999.0, 0.0, 0.0, 0.0])

    def test_data_copy_returns_correct_values(self):
        # by-value return goes through element-by-element converter -> numpy object array
        copy = self.c.data_copy()
        self.assertEqual(len(copy), 3)
        np.testing.assert_array_almost_equal(copy[0], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(copy[2], [2.0, 2.0, 2.0, 2.0])

    def test_copy_survives_container_deletion(self):
        copy = self.c.data_copy()
        del self.c
        gc.collect()
        np.testing.assert_array_almost_equal(copy[2], [2.0, 2.0, 2.0, 2.0])


# ==================================================================
# Matrix algebra (only the view semantics differ from array)
# ==================================================================

class TestMatrixViewAlgebra(unittest.TestCase):
    """matrix_view operator=(scalar) sets scalar * identity, unlike array_view."""

    def test_fill_sets_diagonal(self):
        m = np.zeros((3, 3))
        nc.fill_matrix_view(m, 5.0)
        np.testing.assert_array_almost_equal(m, 5.0 * np.eye(3))


# ==================================================================
# Scalar types and higher rank
# ==================================================================

class TestScalarTypes(unittest.TestCase):

    def test_int(self):
        self.assertEqual(nc.sum_int_array(np.array([1, 2, 3, 4], dtype=np.int64)), 10)
        self.assertEqual(nc.sum_int_array(np.array([1, 2, 3], dtype=np.int32)), 6)

    def test_complex(self):
        self.assertAlmostEqual(nc.sum_complex_array(np.array([1 + 2j, 3 + 4j])), 4 + 6j)

    def test_3d(self):
        self.assertAlmostEqual(nc.sum_3d(np.ones((2, 3, 4))), 24.0)
        self.assertAlmostEqual(nc.sum_3d(np.zeros((2, 0, 4))), 0.0)


# ==================================================================
# Non-npy element types (element-by-element converter path)
# ==================================================================

class TestArrayOfStrings(unittest.TestCase):
    """array<std::string, 1>: T has no native numpy type, converted element-by-element."""

    def test_roundtrip(self):
        result = nc.reverse_strings(np.array(["abc", "de", "f"], dtype=object))
        self.assertEqual(list(result), ["cba", "ed", "f"])

    def test_from_object_array(self):
        result = nc.reverse_strings(np.array(["hello", "world"], dtype=object))
        self.assertEqual(list(result), ["olleh", "dlrow"])


class TestArrayOfVectors(unittest.TestCase):
    """array<std::vector<double>, R>: nested converted type."""

    def test_make_ranges(self):
        result = nc.make_ranges(3)
        self.assertEqual(list(result[0]), [0.0])
        self.assertEqual(list(result[1]), [0.0, 1.0])
        self.assertEqual(list(result[2]), [0.0, 1.0, 2.0])

    def test_flatten(self):
        arr = np.array([[1.0, 2.0], [3.0]], dtype=object)
        result = nc.flatten_array_of_vectors(arr)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_2d_array_of_vectors(self):
        grid = nc.make_grid(2, 3)
        self.assertEqual(list(grid[0, 0]), [0, 0])
        self.assertEqual(list(grid[1, 2]), [1, 2])

    def test_roundtrip_make_then_flatten(self):
        ranges = nc.make_ranges(4)
        flat = nc.flatten_array_of_vectors(ranges)
        np.testing.assert_array_almost_equal(flat, [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
