# Build google benchmark
add_subdirectory(google_benchmark EXCLUDE_FROM_ALL)

# The list of benchs
set(all_benchs view iterator_foreach loop1d loop2d loop3d)

foreach(bench ${all_benchs})
  get_filename_component(bench_name ${bench} NAME_WE)
  add_executable(${bench_name} ${bench})
  target_link_libraries(${bench_name} nda_c benchmark_main project_warnings)
endforeach()
