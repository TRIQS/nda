target_link_libraries(${PROJECT_NAME}_c PUBLIC mpi::mpi_c)
target_include_directories(${PROJECT_NAME}_c PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)