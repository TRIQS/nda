target_link_libraries(${PROJECT_NAME}_c PUBLIC mpi::mpi_c)

target_compile_definitions(${PROJECT_NAME}_c PRIVATE
        $<BUILD_INTERFACE:MPI_SUPPORT>
        $<INSTALL_INTERFACE:MPI_SUPPORT>
)