/*
 @licstart  The following is the entire license notice for the JavaScript code in this file.

 The MIT License (MIT)

 Copyright (C) 1997-2020 by Dimitri van Heesch

 Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 and associated documentation files (the "Software"), to deal in the Software without restriction,
 including without limitation the rights to use, copy, modify, merge, publish, distribute,
 sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all copies or
 substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

 @licend  The above is the entire license notice for the JavaScript code in this file
*/
var NAVTREE =
[
  [ "TRIQS/nda", "index.html", [
    [ "Overview", "index.html", "index" ],
    [ "Installation", "installation.html", [
      [ "Dependencies", "installation.html#dependencies", null ],
      [ "Installation steps", "installation.html#install_steps", null ],
      [ "Versions", "installation.html#versions", null ],
      [ "Custom CMake options", "installation.html#cmake_options", null ]
    ] ],
    [ "Integration in C++ projects", "integration.html", [
      [ "CMake", "integration.html#cmake", [
        [ "FetchContent", "integration.html#fetch", null ],
        [ "find_package", "integration.html#find_package", null ],
        [ "add_subdirectory", "integration.html#add_sub", null ]
      ] ]
    ] ],
    [ "Examples", "examples.html", [
      [ "Example 1: A quick overview", "ex1.html", null ],
      [ "Example 2: Constructing arrays", "ex2.html", null ],
      [ "Example 3: Initializing arrays", "ex3.html", null ],
      [ "Example 4: Views and slices", "ex4.html", null ],
      [ "Example 5: HDF5 support", "ex5.html", null ],
      [ "Example 6: MPI support", "ex6.html", null ],
      [ "Example 7: Making use of symmetries", "ex7.html", null ],
      [ "Example 8: Linear algebra support", "ex8.html", null ]
    ] ],
    [ "API Documentation", "documentation.html", [
      [ "Arrays and views", "group__arrays__views.html", [
        [ "basic_array", "classnda_1_1basic__array.html", null ],
        [ "basic_array_view", "classnda_1_1basic__array__view.html", null ],
        [ "Algorithms", "group__av__algs.html", null ],
        [ "Arithmetic operations", "group__av__ops.html", [
          [ "expr", "structnda_1_1expr.html", null ],
          [ "expr_unary", "structnda_1_1expr__unary.html", null ]
        ] ],
        [ "Array/View utilities", "group__av__utils.html", [
          [ "Array", "conceptnda_1_1_array.html", null ],
          [ "MemoryArray", "conceptnda_1_1_memory_array.html", null ],
          [ "ArrayOfRank", "conceptnda_1_1_array_of_rank.html", null ],
          [ "MemoryArrayOfRank", "conceptnda_1_1_memory_array_of_rank.html", null ],
          [ "ArrayOrScalar", "conceptnda_1_1_array_or_scalar.html", null ],
          [ "Matrix", "conceptnda_1_1_matrix.html", null ],
          [ "Vector", "conceptnda_1_1_vector.html", null ],
          [ "MemoryMatrix", "conceptnda_1_1_memory_matrix.html", null ],
          [ "MemoryVector", "conceptnda_1_1_memory_vector.html", null ],
          [ "ArrayInitializer", "conceptnda_1_1_array_initializer.html", null ],
          [ "HasValueTypeConstructibleFrom", "conceptnda_1_1_has_value_type_constructible_from.html", null ],
          [ "array_adapter", "classnda_1_1array__adapter.html", null ],
          [ "array_iterator<Rank, T, Pointer>", "classnda_1_1array__iterator.html", null ],
          [ "array_iterator<1, T, Pointer>", "classnda_1_1array__iterator_3_011_00_01_t_00_01_pointer_01_4.html", null ],
          [ "default_accessor", "structnda_1_1default__accessor.html", null ],
          [ "default_accessor::accessor", "structnda_1_1default__accessor_1_1accessor.html", null ],
          [ "no_alias_accessor", "structnda_1_1no__alias__accessor.html", null ],
          [ "no_alias_accessor::accessor", "structnda_1_1no__alias__accessor_1_1accessor.html", null ]
        ] ],
        [ "Factories and transformations", "group__av__factories.html", null ],
        [ "HDF5 support", "group__av__hdf5.html", null ],
        [ "MPI support", "group__av__mpi.html", null ],
        [ "Mathematical functions", "group__av__math.html", [
          [ "expr_call", "structnda_1_1expr__call.html", null ],
          [ "mapped", "structnda_1_1mapped.html", null ]
        ] ],
        [ "Symmetries", "group__av__sym.html", [
          [ "NdaSymmetry", "conceptnda_1_1_nda_symmetry.html", null ],
          [ "NdaInitFunc", "conceptnda_1_1_nda_init_func.html", null ],
          [ "operation", "structnda_1_1operation.html", null ],
          [ "sym_grp", "classnda_1_1sym__grp.html", null ]
        ] ],
        [ "Typedefs", "group__av__types.html", null ]
      ] ],
      [ "Compile-time lazy expressions and functions", "group__clef.html", [
        [ "Automatic assignment", "group__clef__autoassign.html", null ],
        [ "CLEF utilities", "group__clef__utils.html", null ],
        [ "Evaluation of lazy objects", "group__clef__eval.html", [
          [ "clef::evaluator", "structnda_1_1clef_1_1evaluator.html", null ],
          [ "clef::evaluator<expr<Tag,Childs...>, Pairs...>", "structnda_1_1clef_1_1evaluator_3_01expr_3_01_tag_00_01_childs_8_8_8_01_4_00_01_pairs_8_8_8_01_4.html", null ],
          [ "clef::evaluator<make_fun_impl<T,Is...>,Pairs...>", "structnda_1_1clef_1_1evaluator_3_01make__fun__impl_3_01_t_00_01_is_8_8_8_01_4_00_01_pairs_8_8_8_01_4.html", null ],
          [ "clef::evaluator<placeholder<N>, pair<Is,Ts >...>", "structnda_1_1clef_1_1evaluator_3_01placeholder_3_01_n_01_4_00_01pair_3_01_is_00_01_ts_01_4_8_8_8_01_4.html", null ],
          [ "clef::evaluator<std::reference_wrapper<T>, Pairs...>", "structnda_1_1clef_1_1evaluator_3_01std_1_1reference__wrapper_3_01_t_01_4_00_01_pairs_8_8_8_01_4.html", null ]
        ] ],
        [ "Lazy expressions, functions and operations", "group__clef__expr.html", [
          [ "clef::expr", "structnda_1_1clef_1_1expr.html", null ],
          [ "clef::make_fun_impl", "structnda_1_1clef_1_1make__fun__impl.html", null ],
          [ "clef::operation", "structnda_1_1clef_1_1operation.html", null ],
          [ "clef::operation<tags::divides>", "structnda_1_1clef_1_1operation_3_01tags_1_1divides_01_4.html", null ],
          [ "clef::operation<tags::eq>", "structnda_1_1clef_1_1operation_3_01tags_1_1eq_01_4.html", null ],
          [ "clef::operation<tags::function>", "structnda_1_1clef_1_1operation_3_01tags_1_1function_01_4.html", null ],
          [ "clef::operation<tags::geq>", "structnda_1_1clef_1_1operation_3_01tags_1_1geq_01_4.html", null ],
          [ "clef::operation<tags::greater>", "structnda_1_1clef_1_1operation_3_01tags_1_1greater_01_4.html", null ],
          [ "clef::operation<tags::if_else>", "structnda_1_1clef_1_1operation_3_01tags_1_1if__else_01_4.html", null ],
          [ "clef::operation<tags::leq>", "structnda_1_1clef_1_1operation_3_01tags_1_1leq_01_4.html", null ],
          [ "clef::operation<tags::less>", "structnda_1_1clef_1_1operation_3_01tags_1_1less_01_4.html", null ],
          [ "clef::operation<tags::loginot>", "structnda_1_1clef_1_1operation_3_01tags_1_1loginot_01_4.html", null ],
          [ "clef::operation<tags::minus>", "structnda_1_1clef_1_1operation_3_01tags_1_1minus_01_4.html", null ],
          [ "clef::operation<tags::multiplies>", "structnda_1_1clef_1_1operation_3_01tags_1_1multiplies_01_4.html", null ],
          [ "clef::operation<tags::negate>", "structnda_1_1clef_1_1operation_3_01tags_1_1negate_01_4.html", null ],
          [ "clef::operation<tags::plus>", "structnda_1_1clef_1_1operation_3_01tags_1_1plus_01_4.html", null ],
          [ "clef::operation<tags::subscript>", "structnda_1_1clef_1_1operation_3_01tags_1_1subscript_01_4.html", null ],
          [ "clef::operation<tags::terminal>", "structnda_1_1clef_1_1operation_3_01tags_1_1terminal_01_4.html", null ],
          [ "clef::operation<tags::unaryplus>", "structnda_1_1clef_1_1operation_3_01tags_1_1unaryplus_01_4.html", null ],
          [ "clef::tags::binary_op", "structnda_1_1clef_1_1tags_1_1binary__op.html", null ],
          [ "clef::tags::divides", "structnda_1_1clef_1_1tags_1_1divides.html", null ],
          [ "clef::tags::eq", "structnda_1_1clef_1_1tags_1_1eq.html", null ],
          [ "clef::tags::function", "structnda_1_1clef_1_1tags_1_1function.html", null ],
          [ "clef::tags::geq", "structnda_1_1clef_1_1tags_1_1geq.html", null ],
          [ "clef::tags::greater", "structnda_1_1clef_1_1tags_1_1greater.html", null ],
          [ "clef::tags::if_else", "structnda_1_1clef_1_1tags_1_1if__else.html", null ],
          [ "clef::tags::leq", "structnda_1_1clef_1_1tags_1_1leq.html", null ],
          [ "clef::tags::less", "structnda_1_1clef_1_1tags_1_1less.html", null ],
          [ "clef::tags::loginot", "structnda_1_1clef_1_1tags_1_1loginot.html", null ],
          [ "clef::tags::minus", "structnda_1_1clef_1_1tags_1_1minus.html", null ],
          [ "clef::tags::multiplies", "structnda_1_1clef_1_1tags_1_1multiplies.html", null ],
          [ "clef::tags::negate", "structnda_1_1clef_1_1tags_1_1negate.html", null ],
          [ "clef::tags::plus", "structnda_1_1clef_1_1tags_1_1plus.html", null ],
          [ "clef::tags::subscript", "structnda_1_1clef_1_1tags_1_1subscript.html", null ],
          [ "clef::tags::terminal", "structnda_1_1clef_1_1tags_1_1terminal.html", null ],
          [ "clef::tags::unary_op", "structnda_1_1clef_1_1tags_1_1unary__op.html", null ],
          [ "clef::tags::unaryplus", "structnda_1_1clef_1_1tags_1_1unaryplus.html", null ]
        ] ],
        [ "Placeholders", "group__clef__placeholders.html", [
          [ "clef::pair", "structnda_1_1clef_1_1pair.html", null ],
          [ "clef::placeholder", "structnda_1_1clef_1_1placeholder.html", null ]
        ] ]
      ] ],
      [ "Linear algebra", "group__linalg.html", [
        [ "BLAS interface", "group__linalg__blas.html", null ],
        [ "BLAS utilities", "group__linalg__blas__utils.html", null ],
        [ "LAPACK interface", "group__linalg__lapack.html", [
          [ "lapack::gelss_worker", "classnda_1_1lapack_1_1gelss__worker.html", null ],
          [ "lapack::gelss_worker_hermitian", "classnda_1_1lapack_1_1gelss__worker__hermitian.html", null ]
        ] ],
        [ "Linear algebra tools", "group__linalg__tools.html", null ]
      ] ],
      [ "Memory layout", "group__layout.html", [
        [ "Layout policies", "group__layout__pols.html", [
          [ "basic_layout", "structnda_1_1basic__layout.html", null ],
          [ "C_layout", "structnda_1_1_c__layout.html", null ],
          [ "C_stride_layout", "structnda_1_1_c__stride__layout.html", null ],
          [ "F_layout", "structnda_1_1_f__layout.html", null ],
          [ "F_stride_layout", "structnda_1_1_f__stride__layout.html", null ]
        ] ],
        [ "Layout utilities", "group__layout__utils.html", [
          [ "_linear_index_t", "structnda_1_1__linear__index__t.html", null ],
          [ "ellipsis", "structnda_1_1ellipsis.html", null ],
          [ "idx_group_t", "structnda_1_1idx__group__t.html", null ],
          [ "layout_info_t", "structnda_1_1layout__info__t.html", null ]
        ] ],
        [ "Mult-dimensional indexing", "group__layout__idx.html", [
          [ "idx_map", "classnda_1_1idx__map.html", null ]
        ] ]
      ] ],
      [ "Memory management", "group__memory.html", [
        [ "Address spaces", "group__mem__addrspcs.html", null ],
        [ "Allocators", "group__mem__allocators.html", [
          [ "mem::blk_t", "structnda_1_1mem_1_1blk__t.html", null ],
          [ "mem::bucket", "classnda_1_1mem_1_1bucket.html", null ],
          [ "mem::leak_check", "classnda_1_1mem_1_1leak__check.html", null ],
          [ "mem::mallocator", "classnda_1_1mem_1_1mallocator.html", null ],
          [ "mem::multi_bucket", "classnda_1_1mem_1_1multi__bucket.html", null ],
          [ "mem::segregator", "classnda_1_1mem_1_1segregator.html", null ],
          [ "mem::stats", "classnda_1_1mem_1_1stats.html", null ]
        ] ],
        [ "Handles", "group__mem__handles.html", [
          [ "mem::handle_borrowed", "structnda_1_1mem_1_1handle__borrowed.html", null ],
          [ "mem::handle_heap", "structnda_1_1mem_1_1handle__heap.html", null ],
          [ "mem::handle_shared", "structnda_1_1mem_1_1handle__shared.html", null ],
          [ "mem::handle_sso", "structnda_1_1mem_1_1handle__sso.html", null ],
          [ "mem::handle_stack", "structnda_1_1mem_1_1handle__stack.html", null ]
        ] ],
        [ "Memory policies", "group__mem__pols.html", [
          [ "borrowed", "structnda_1_1borrowed.html", null ],
          [ "heap_basic", "structnda_1_1heap__basic.html", null ],
          [ "shared", "structnda_1_1shared.html", null ],
          [ "sso", "structnda_1_1sso.html", null ],
          [ "stack", "structnda_1_1stack.html", null ]
        ] ],
        [ "Memory utilities", "group__mem__utils.html", [
          [ "mem::Allocator", "conceptnda_1_1mem_1_1_allocator.html", null ],
          [ "mem::Handle", "conceptnda_1_1mem_1_1_handle.html", null ],
          [ "mem::OwningHandle", "conceptnda_1_1mem_1_1_owning_handle.html", null ],
          [ "mem::aligner", "structnda_1_1mem_1_1aligner.html", null ],
          [ "mem::do_not_initialize_t", "structnda_1_1mem_1_1do__not__initialize__t.html", null ],
          [ "mem::init_zero_t", "structnda_1_1mem_1_1init__zero__t.html", null ]
        ] ]
      ] ],
      [ "Testing tools", "group__testing.html", null ],
      [ "Utilities", "group__utilities.html", [
        [ "Concepts", "group__utils__concepts.html", [
          [ "CallableWithLongs", "conceptnda_1_1_callable_with_longs.html", null ],
          [ "StdArrayOfLong", "conceptnda_1_1_std_array_of_long.html", null ],
          [ "Scalar", "conceptnda_1_1_scalar.html", null ],
          [ "DoubleOrComplex", "conceptnda_1_1_double_or_complex.html", null ],
          [ "InstantiationOf", "conceptnda_1_1_instantiation_of.html", null ]
        ] ],
        [ "Extensions to the standard library", "group__utils__std.html", [
          [ "runtime_error", "classnda_1_1runtime__error.html", null ]
        ] ],
        [ "Permutations", "group__utils__perms.html", null ],
        [ "Type traits", "group__utils__type__traits.html", [
          [ "is_instantiation_of", "structnda_1_1is__instantiation__of.html", null ]
        ] ]
      ] ],
      [ "File List", "files.html", "files" ]
    ] ],
    [ "Changelog", "changelog.html", null ],
    [ "Issues", "issues.html", null ]
  ] ]
];

var NAVTREEINDEX =
[
"__impl__basic__array__view__common_8hpp_source.html",
"layout_2policies_8hpp.html"
];

var SYNCONMSG = 'click to disable panel synchronization';
var SYNCOFFMSG = 'click to enable panel synchronization';
var LISTOFALLMEMBERS = 'List of all members';