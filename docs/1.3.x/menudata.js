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
var menudata={children:[
{text:"Main Page",url:"index.html"},
{text:"Installation",url:"installation.html",children:[
{text:"Dependencies",url:"installation.html#dependencies"},
{text:"Installation steps",url:"installation.html#install_steps"},
{text:"Versions",url:"installation.html#versions"},
{text:"Custom CMake options",url:"installation.html#cmake_options"}]},
{text:"Integration in C++ projects",url:"integration.html",children:[
{text:"CMake",url:"integration.html#cmake",children:[
{text:"FetchContent",url:"integration.html#fetch"},
{text:"find_package",url:"integration.html#find_package"},
{text:"add_subdirectory",url:"integration.html#add_sub"}]}]},
{text:"Examples",url:"examples.html"},
{text:"API Documentation",url:"documentation.html",children:[
{text:"Arrays and views",url:"group__arrays__views.html",children:[
{text:"basic_array",url:"classnda_1_1basic__array.html"},
{text:"basic_array_view",url:"classnda_1_1basic__array__view.html"},
{text:"Algorithms",url:"group__av__algs.html"},
{text:"Arithmetic operations",url:"group__av__ops.html",children:[
{text:"expr",url:"structnda_1_1expr.html"},
{text:"expr_unary",url:"structnda_1_1expr__unary.html"}]},
{text:"Array/View utilities",url:"group__av__utils.html",children:[
{text:"Array",url:"conceptnda_1_1_array.html"},
{text:"MemoryArray",url:"conceptnda_1_1_memory_array.html"},
{text:"ArrayOfRank",url:"conceptnda_1_1_array_of_rank.html"},
{text:"MemoryArrayOfRank",url:"conceptnda_1_1_memory_array_of_rank.html"},
{text:"ArrayOrScalar",url:"conceptnda_1_1_array_or_scalar.html"},
{text:"Matrix",url:"conceptnda_1_1_matrix.html"},
{text:"Vector",url:"conceptnda_1_1_vector.html"},
{text:"MemoryMatrix",url:"conceptnda_1_1_memory_matrix.html"},
{text:"MemoryVector",url:"conceptnda_1_1_memory_vector.html"},
{text:"ArrayInitializer",url:"conceptnda_1_1_array_initializer.html"},
{text:"HasValueTypeConstructibleFrom",url:"conceptnda_1_1_has_value_type_constructible_from.html"},
{text:"array_adapter",url:"classnda_1_1array__adapter.html"},
{text:"array_iterator<Rank, T, Pointer>",url:"classnda_1_1array__iterator.html"},
{text:"array_iterator<1, T, Pointer>",url:"classnda_1_1array__iterator_3_011_00_01_t_00_01_pointer_01_4.html"},
{text:"default_accessor",url:"structnda_1_1default__accessor.html"},
{text:"default_accessor::accessor",url:"structnda_1_1default__accessor_1_1accessor.html"},
{text:"no_alias_accessor",url:"structnda_1_1no__alias__accessor.html"},
{text:"no_alias_accessor::accessor",url:"structnda_1_1no__alias__accessor_1_1accessor.html"}]},
{text:"Factories and transformations",url:"group__av__factories.html"},
{text:"HDF5 support",url:"group__av__hdf5.html"},
{text:"MPI support",url:"group__av__mpi.html",children:[
{text:"mpi::lazy<mpi::tag::gather, A>",url:"structmpi_1_1lazy_3_01mpi_1_1tag_1_1gather_00_01_a_01_4.html"},
{text:"mpi::lazy<mpi::tag::reduce, A>",url:"structmpi_1_1lazy_3_01mpi_1_1tag_1_1reduce_00_01_a_01_4.html"},
{text:"mpi::lazy<mpi::tag::scatter, A>",url:"structmpi_1_1lazy_3_01mpi_1_1tag_1_1scatter_00_01_a_01_4.html"}]},
{text:"Mathematical functions",url:"group__av__math.html",children:[
{text:"conj_f",url:"structnda_1_1conj__f.html"},
{text:"expr_call",url:"structnda_1_1expr__call.html"},
{text:"mapped",url:"structnda_1_1mapped.html"}]},
{text:"Symmetries",url:"group__av__sym.html",children:[
{text:"NdaSymmetry",url:"conceptnda_1_1_nda_symmetry.html"},
{text:"NdaInitFunc",url:"conceptnda_1_1_nda_init_func.html"},
{text:"operation",url:"structnda_1_1operation.html"},
{text:"sym_grp",url:"classnda_1_1sym__grp.html"}]},
{text:"Typedefs",url:"group__av__types.html"}]},
{text:"Compile-time lazy expressions and functions",url:"group__clef.html",children:[
{text:"Automatic assignment",url:"group__clef__autoassign.html"},
{text:"CLEF utilities",url:"group__clef__utils.html"},
{text:"Evaluation of lazy objects",url:"group__clef__eval.html",children:[
{text:"clef::evaluator",url:"structnda_1_1clef_1_1evaluator.html"},
{text:"clef::evaluator<expr<Tag,Childs...>, Pairs...>",url:"structnda_1_1clef_1_1evaluator_3_01expr_3_01_tag_00_01_childs_8_8_8_01_4_00_01_pairs_8_8_8_01_4.html"},
{text:"clef::evaluator<make_fun_impl<T,Is...>,Pairs...>",url:"structnda_1_1clef_1_1evaluator_3_01make__fun__impl_3_01_t_00_01_is_8_8_8_01_4_00_01_pairs_8_8_8_01_4.html"},
{text:"clef::evaluator<placeholder<N>, pair<Is,Ts >...>",url:"structnda_1_1clef_1_1evaluator_3_01placeholder_3_01_n_01_4_00_01pair_3_01_is_00_01_ts_01_4_8_8_8_01_4.html"},
{text:"clef::evaluator<std::reference_wrapper<T>, Pairs...>",url:"structnda_1_1clef_1_1evaluator_3_01std_1_1reference__wrapper_3_01_t_01_4_00_01_pairs_8_8_8_01_4.html"}]},
{text:"Lazy expressions, functions and operations",url:"group__clef__expr.html",children:[
{text:"clef::expr",url:"structnda_1_1clef_1_1expr.html"},
{text:"clef::make_fun_impl",url:"structnda_1_1clef_1_1make__fun__impl.html"},
{text:"clef::operation",url:"structnda_1_1clef_1_1operation.html"},
{text:"clef::operation<tags::divides>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1divides_01_4.html"},
{text:"clef::operation<tags::eq>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1eq_01_4.html"},
{text:"clef::operation<tags::function>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1function_01_4.html"},
{text:"clef::operation<tags::geq>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1geq_01_4.html"},
{text:"clef::operation<tags::greater>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1greater_01_4.html"},
{text:"clef::operation<tags::if_else>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1if__else_01_4.html"},
{text:"clef::operation<tags::leq>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1leq_01_4.html"},
{text:"clef::operation<tags::less>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1less_01_4.html"},
{text:"clef::operation<tags::loginot>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1loginot_01_4.html"},
{text:"clef::operation<tags::minus>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1minus_01_4.html"},
{text:"clef::operation<tags::multiplies>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1multiplies_01_4.html"},
{text:"clef::operation<tags::negate>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1negate_01_4.html"},
{text:"clef::operation<tags::plus>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1plus_01_4.html"},
{text:"clef::operation<tags::subscript>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1subscript_01_4.html"},
{text:"clef::operation<tags::terminal>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1terminal_01_4.html"},
{text:"clef::operation<tags::unaryplus>",url:"structnda_1_1clef_1_1operation_3_01tags_1_1unaryplus_01_4.html"},
{text:"clef::tags::binary_op",url:"structnda_1_1clef_1_1tags_1_1binary__op.html"},
{text:"clef::tags::divides",url:"structnda_1_1clef_1_1tags_1_1divides.html"},
{text:"clef::tags::eq",url:"structnda_1_1clef_1_1tags_1_1eq.html"},
{text:"clef::tags::function",url:"structnda_1_1clef_1_1tags_1_1function.html"},
{text:"clef::tags::geq",url:"structnda_1_1clef_1_1tags_1_1geq.html"},
{text:"clef::tags::greater",url:"structnda_1_1clef_1_1tags_1_1greater.html"},
{text:"clef::tags::if_else",url:"structnda_1_1clef_1_1tags_1_1if__else.html"},
{text:"clef::tags::leq",url:"structnda_1_1clef_1_1tags_1_1leq.html"},
{text:"clef::tags::less",url:"structnda_1_1clef_1_1tags_1_1less.html"},
{text:"clef::tags::loginot",url:"structnda_1_1clef_1_1tags_1_1loginot.html"},
{text:"clef::tags::minus",url:"structnda_1_1clef_1_1tags_1_1minus.html"},
{text:"clef::tags::multiplies",url:"structnda_1_1clef_1_1tags_1_1multiplies.html"},
{text:"clef::tags::negate",url:"structnda_1_1clef_1_1tags_1_1negate.html"},
{text:"clef::tags::plus",url:"structnda_1_1clef_1_1tags_1_1plus.html"},
{text:"clef::tags::subscript",url:"structnda_1_1clef_1_1tags_1_1subscript.html"},
{text:"clef::tags::terminal",url:"structnda_1_1clef_1_1tags_1_1terminal.html"},
{text:"clef::tags::unary_op",url:"structnda_1_1clef_1_1tags_1_1unary__op.html"},
{text:"clef::tags::unaryplus",url:"structnda_1_1clef_1_1tags_1_1unaryplus.html"}]},
{text:"Placeholders",url:"group__clef__placeholders.html",children:[
{text:"clef::pair",url:"structnda_1_1clef_1_1pair.html"},
{text:"clef::placeholder",url:"structnda_1_1clef_1_1placeholder.html"}]}]},
{text:"Linear algebra",url:"group__linalg.html",children:[
{text:"BLAS interface",url:"group__linalg__blas.html"},
{text:"LAPACK interface",url:"group__linalg__lapack.html",children:[
{text:"lapack::gelss_worker",url:"classnda_1_1lapack_1_1gelss__worker.html"},
{text:"lapack::gelss_worker_hermitian",url:"structnda_1_1lapack_1_1gelss__worker__hermitian.html"}]},
{text:"Linear algebra tools",url:"group__linalg__tools.html"}]},
{text:"Memory layout",url:"group__layout.html",children:[
{text:"Layout policies",url:"group__layout__pols.html",children:[
{text:"basic_layout",url:"structnda_1_1basic__layout.html"},
{text:"basic_layout_str",url:"structnda_1_1basic__layout__str.html"},
{text:"C_layout",url:"structnda_1_1_c__layout.html"},
{text:"C_layout_str",url:"structnda_1_1_c__layout__str.html"},
{text:"C_stride_layout",url:"structnda_1_1_c__stride__layout.html"},
{text:"C_stride_layout_str",url:"structnda_1_1_c__stride__layout__str.html"},
{text:"F_layout",url:"structnda_1_1_f__layout.html"},
{text:"F_layout_str",url:"structnda_1_1_f__layout__str.html"},
{text:"F_stride_layout",url:"structnda_1_1_f__stride__layout.html"},
{text:"F_stride_layout_str",url:"structnda_1_1_f__stride__layout__str.html"}]},
{text:"Layout utilities",url:"group__layout__utils.html",children:[
{text:"_linear_index_t",url:"structnda_1_1__linear__index__t.html"},
{text:"ellipsis",url:"structnda_1_1ellipsis.html"},
{text:"idx_group_t",url:"structnda_1_1idx__group__t.html"},
{text:"layout_info_t",url:"structnda_1_1layout__info__t.html"}]},
{text:"Mult-dimensional indexing",url:"group__layout__idx.html",children:[
{text:"idx_map",url:"classnda_1_1idx__map.html"},
{text:"rect_str",url:"classnda_1_1rect__str.html"}]}]},
{text:"Memory management",url:"group__memory.html",children:[
{text:"Address spaces",url:"group__mem__addrspcs.html"},
{text:"Allocators",url:"group__mem__allocators.html",children:[
{text:"mem::blk_t",url:"structnda_1_1mem_1_1blk__t.html"},
{text:"mem::bucket",url:"classnda_1_1mem_1_1bucket.html"},
{text:"mem::leak_check",url:"classnda_1_1mem_1_1leak__check.html"},
{text:"mem::mallocator",url:"classnda_1_1mem_1_1mallocator.html"},
{text:"mem::multi_bucket",url:"classnda_1_1mem_1_1multi__bucket.html"},
{text:"mem::segregator",url:"classnda_1_1mem_1_1segregator.html"},
{text:"mem::stats",url:"classnda_1_1mem_1_1stats.html"}]},
{text:"Handles",url:"group__mem__handles.html",children:[
{text:"mem::handle_borrowed",url:"structnda_1_1mem_1_1handle__borrowed.html"},
{text:"mem::handle_heap",url:"structnda_1_1mem_1_1handle__heap.html"},
{text:"mem::handle_shared",url:"structnda_1_1mem_1_1handle__shared.html"},
{text:"mem::handle_sso",url:"structnda_1_1mem_1_1handle__sso.html"},
{text:"mem::handle_stack",url:"structnda_1_1mem_1_1handle__stack.html"}]},
{text:"Memory policies",url:"group__mem__pols.html",children:[
{text:"borrowed",url:"structnda_1_1borrowed.html"},
{text:"heap_basic",url:"structnda_1_1heap__basic.html"},
{text:"shared",url:"structnda_1_1shared.html"},
{text:"sso",url:"structnda_1_1sso.html"},
{text:"stack",url:"structnda_1_1stack.html"}]},
{text:"Memory utilities",url:"group__mem__utils.html",children:[
{text:"mem::Allocator",url:"conceptnda_1_1mem_1_1_allocator.html"},
{text:"mem::Handle",url:"conceptnda_1_1mem_1_1_handle.html"},
{text:"mem::OwningHandle",url:"conceptnda_1_1mem_1_1_owning_handle.html"},
{text:"mem::aligner",url:"structnda_1_1mem_1_1aligner.html"},
{text:"mem::do_not_initialize_t",url:"structnda_1_1mem_1_1do__not__initialize__t.html"},
{text:"mem::init_zero_t",url:"structnda_1_1mem_1_1init__zero__t.html"}]}]},
{text:"Testing tools",url:"group__testing.html"},
{text:"Utilities",url:"group__utilities.html",children:[
{text:"Concepts",url:"group__utils__concepts.html",children:[
{text:"CallableWithLongs",url:"conceptnda_1_1_callable_with_longs.html"},
{text:"StdArrayOfLong",url:"conceptnda_1_1_std_array_of_long.html"},
{text:"Scalar",url:"conceptnda_1_1_scalar.html"},
{text:"DoubleOrComplex",url:"conceptnda_1_1_double_or_complex.html"},
{text:"InstantiationOf",url:"conceptnda_1_1_instantiation_of.html"}]},
{text:"Extensions to the standard library",url:"group__utils__std.html",children:[
{text:"runtime_error",url:"classnda_1_1runtime__error.html"}]},
{text:"Permutations",url:"group__utils__perms.html"},
{text:"Type traits",url:"group__utils__type__traits.html",children:[
{text:"is_instantiation_of",url:"structnda_1_1is__instantiation__of.html"}]}]},
{text:"File List",url:"files.html"}]},
{text:"Changelog",url:"changelog.html"},
{text:"Issues",url:"issues.html"}]}
