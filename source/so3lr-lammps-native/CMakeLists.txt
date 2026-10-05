cmake_minimum_required(VERSION 3.20)
project(so3lr_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(so3lr_native STATIC src/native_model.cpp src/so3lr_architecture.cpp src/so3lr_charge_spin.cpp
                                src/so3lr_capabilities.cpp)
target_include_directories(so3lr_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(so3lr_native PRIVATE -Wall -Wextra -Wpedantic)

add_executable(so3lr_native_dev2_test tests/test_loader_and_inputs.cpp)
target_link_libraries(so3lr_native_dev2_test PRIVATE so3lr_native)
target_compile_options(so3lr_native_dev2_test PRIVATE -Wall -Wextra -Wpedantic)

# Architecture descriptor: CPU-only, so it is built and tested without Kokkos.
add_executable(so3lr_architecture_test tests/test_so3lr_architecture.cpp)
target_link_libraries(so3lr_architecture_test PRIVATE so3lr_native)
target_compile_options(so3lr_architecture_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(so3lr_degree_layout_test tests/test_degree_layout.cpp)
target_link_libraries(so3lr_degree_layout_test PRIVATE so3lr_native)
target_compile_options(so3lr_degree_layout_test PRIVATE -Wall -Wextra -Wpedantic)

add_executable(so3lr_charge_spin_table tests/so3lr_charge_spin_table.cpp)
target_link_libraries(so3lr_charge_spin_table PRIVATE so3lr_native)
target_compile_options(so3lr_charge_spin_table PRIVATE -Wall -Wextra -Wpedantic)

add_executable(so3lr_capability_probe tests/so3lr_capability_probe.cpp)
target_link_libraries(so3lr_capability_probe PRIVATE so3lr_native)
target_compile_options(so3lr_capability_probe PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
# Needs no model file and no GPU.
add_test(NAME degree_layout COMMAND so3lr_degree_layout_test)
set(SO3LR_TEST_MODEL "" CACHE FILEPATH "Path to a .so3lr model used by the tests")
if(SO3LR_TEST_MODEL)
  add_test(NAME architecture_descriptor
           COMMAND so3lr_architecture_test ${SO3LR_TEST_MODEL})
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME capability_rejection
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_capability_rejection.py
                     $<TARGET_FILE:so3lr_capability_probe> ${SO3LR_TEST_MODEL})
  endif()
endif()

option(SO3LR_ENABLE_KOKKOS "Build native Kokkos/CUDA operators" OFF)
set(SO3LR_KOKKOS_SOURCE "" CACHE PATH "Path to bundled Kokkos source")
if(SO3LR_ENABLE_KOKKOS)
  if(NOT EXISTS "${SO3LR_KOKKOS_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "SO3LR_KOKKOS_SOURCE does not contain Kokkos")
  endif()
  add_subdirectory("${SO3LR_KOKKOS_SOURCE}" "${CMAKE_BINARY_DIR}/kokkos")
  find_package(CUDAToolkit REQUIRED)

  add_library(so3lr_kokkos STATIC src/kokkos_input_ops.cpp)
  target_include_directories(so3lr_kokkos PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_kokkos PUBLIC so3lr_native Kokkos::kokkos)
  target_compile_options(so3lr_kokkos PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_filter0 STATIC src/kokkos_filter_block0.cpp)
  target_include_directories(so3lr_filter0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_filter0 PUBLIC so3lr_kokkos)
  target_compile_options(so3lr_filter0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_filter0_blas STATIC src/kokkos_filter_block0_blas.cpp)
  target_include_directories(so3lr_filter0_blas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_filter0_blas PUBLIC so3lr_filter0 CUDA::cublas)
  target_compile_options(so3lr_filter0_blas PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_qkv0 STATIC src/kokkos_qkv_block0.cpp)
  target_include_directories(so3lr_qkv0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_qkv0 PUBLIC so3lr_filter0_blas CUDA::cublas)
  target_compile_options(so3lr_qkv0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_attention0 STATIC src/kokkos_attention_scatter_block0.cpp)
  target_include_directories(so3lr_attention0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_attention0 PUBLIC so3lr_qkv0)
  target_compile_options(so3lr_attention0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_integrated0 STATIC src/kokkos_integrated_block0.cpp)
  target_include_directories(so3lr_integrated0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_integrated0 PUBLIC so3lr_attention0 CUDA::cublas)
  target_compile_options(so3lr_integrated0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_post_attention0 STATIC src/kokkos_post_attention_block0.cpp)
  target_include_directories(so3lr_post_attention0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_post_attention0 PUBLIC so3lr_integrated0 CUDA::cublas)
  target_compile_options(so3lr_post_attention0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_transformer0 STATIC src/kokkos_transformer_block0.cpp)
  target_include_directories(so3lr_transformer0 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_transformer0 PUBLIC so3lr_post_attention0)
  target_compile_options(so3lr_transformer0 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_feature_transformer STATIC src/kokkos_feature_transformer_block.cpp)
  target_include_directories(so3lr_feature_transformer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_feature_transformer PUBLIC so3lr_transformer0)
  target_compile_options(so3lr_feature_transformer PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_transformer_stack01 STATIC src/kokkos_transformer_stack01.cpp)
  target_include_directories(so3lr_transformer_stack01 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_transformer_stack01 PUBLIC so3lr_feature_transformer)
  target_compile_options(so3lr_transformer_stack01 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_transformer_stack012 STATIC src/kokkos_transformer_stack012.cpp)
  target_include_directories(so3lr_transformer_stack012 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_transformer_stack012 PUBLIC so3lr_transformer_stack01)
  target_compile_options(so3lr_transformer_stack012 PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_output_heads STATIC src/kokkos_output_heads.cpp)
  target_include_directories(so3lr_output_heads PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_output_heads PUBLIC so3lr_transformer_stack012 CUDA::cublas)
  target_compile_options(so3lr_output_heads PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_forward_model STATIC src/kokkos_forward_model.cpp)
  target_include_directories(so3lr_forward_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_forward_model PUBLIC so3lr_output_heads)
  target_compile_options(so3lr_forward_model PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_post_attention_reverse STATIC src/kokkos_post_attention_reverse.cpp)
  target_include_directories(so3lr_post_attention_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_post_attention_reverse PUBLIC so3lr_post_attention0 CUDA::cublas)
  target_compile_options(so3lr_post_attention_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_attention_scatter_reverse STATIC src/kokkos_attention_scatter_reverse.cpp)
  target_include_directories(so3lr_attention_scatter_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_attention_scatter_reverse PUBLIC so3lr_kokkos)
  target_compile_options(so3lr_attention_scatter_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_block2_attention_input_reverse STATIC src/kokkos_block2_attention_input_reverse.cpp)
  target_include_directories(so3lr_block2_attention_input_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_block2_attention_input_reverse PUBLIC so3lr_attention_scatter_reverse so3lr_integrated0 CUDA::cublas)
  target_compile_options(so3lr_block2_attention_input_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_complete_transformer_reverse STATIC src/kokkos_complete_transformer_reverse.cpp)
  target_include_directories(so3lr_complete_transformer_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_complete_transformer_reverse PUBLIC so3lr_block2_attention_input_reverse so3lr_post_attention_reverse)
  target_compile_options(so3lr_complete_transformer_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_learned_energy_reverse STATIC src/kokkos_learned_energy_reverse.cpp)
  target_include_directories(so3lr_learned_energy_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_learned_energy_reverse PUBLIC so3lr_complete_transformer_reverse so3lr_output_heads)
  target_compile_options(so3lr_learned_energy_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_learned_energy_forces STATIC src/kokkos_learned_energy_forces.cpp)
  target_include_directories(so3lr_learned_energy_forces PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_learned_energy_forces PUBLIC so3lr_learned_energy_reverse)
  target_compile_options(so3lr_learned_energy_forces PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev3_gpu_test tests/test_kokkos_input_ops.cpp)
  target_link_libraries(so3lr_native_dev3_gpu_test PRIVATE so3lr_kokkos)
  target_compile_options(so3lr_native_dev3_gpu_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev4_filter_test tests/test_kokkos_filter_block0.cpp)
  target_link_libraries(so3lr_native_dev4_filter_test PRIVATE so3lr_filter0)
  target_compile_options(so3lr_native_dev4_filter_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev5_blas_test tests/test_kokkos_filter_block0_blas.cpp)
  target_link_libraries(so3lr_native_dev5_blas_test PRIVATE so3lr_filter0_blas)
  target_compile_options(so3lr_native_dev5_blas_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev6_qkv_test tests/test_kokkos_qkv_block0.cpp)
  target_link_libraries(so3lr_native_dev6_qkv_test PRIVATE so3lr_qkv0)
  target_compile_options(so3lr_native_dev6_qkv_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev7_attention_test tests/test_kokkos_attention_scatter_block0.cpp)
  target_link_libraries(so3lr_native_dev7_attention_test PRIVATE so3lr_attention0)
  target_compile_options(so3lr_native_dev7_attention_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev8_pipeline_test tests/test_kokkos_integrated_block0.cpp)
  target_link_libraries(so3lr_native_dev8_pipeline_test PRIVATE so3lr_integrated0)
  target_compile_options(so3lr_native_dev8_pipeline_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev9_post_attention_test tests/test_kokkos_post_attention_block0.cpp)
  target_link_libraries(so3lr_native_dev9_post_attention_test PRIVATE so3lr_post_attention0)
  target_compile_options(so3lr_native_dev9_post_attention_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev10_transformer_block0_test tests/test_kokkos_transformer_block0.cpp)
  target_link_libraries(so3lr_native_dev10_transformer_block0_test PRIVATE so3lr_transformer0)
  target_compile_options(so3lr_native_dev10_transformer_block0_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev11_feature_transformer_test tests/test_kokkos_feature_transformer_block.cpp)
  target_link_libraries(so3lr_native_dev11_feature_transformer_test PRIVATE so3lr_feature_transformer)
  target_compile_options(so3lr_native_dev11_feature_transformer_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev12_transformer_stack01_test tests/test_kokkos_transformer_stack01.cpp)
  target_link_libraries(so3lr_native_dev12_transformer_stack01_test PRIVATE so3lr_transformer_stack01)
  target_compile_options(so3lr_native_dev12_transformer_stack01_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev13_transformer_stack012_test tests/test_kokkos_transformer_stack012.cpp)
  target_link_libraries(so3lr_native_dev13_transformer_stack012_test PRIVATE so3lr_transformer_stack012)
  target_compile_options(so3lr_native_dev13_transformer_stack012_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev14_output_heads_test tests/test_kokkos_output_heads.cpp)
  target_link_libraries(so3lr_native_dev14_output_heads_test PRIVATE so3lr_output_heads)
  target_compile_options(so3lr_native_dev14_output_heads_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev15_forward_test tests/test_kokkos_forward_model.cpp)
  target_link_libraries(so3lr_native_dev15_forward_test PRIVATE so3lr_forward_model)
  target_compile_options(so3lr_native_dev15_forward_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev16_energy_reverse_test tests/test_kokkos_energy_reverse.cpp)
  target_link_libraries(so3lr_native_dev16_energy_reverse_test PRIVATE so3lr_forward_model)
  target_compile_options(so3lr_native_dev16_energy_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev17_post_attention_reverse_test tests/test_kokkos_post_attention_reverse.cpp)
  target_link_libraries(so3lr_native_dev17_post_attention_reverse_test PRIVATE so3lr_post_attention_reverse so3lr_output_heads)
  target_compile_options(so3lr_native_dev17_post_attention_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev18_attention_scatter_reverse_test tests/test_kokkos_attention_scatter_reverse.cpp)
  target_link_libraries(so3lr_native_dev18_attention_scatter_reverse_test PRIVATE so3lr_attention_scatter_reverse)
  target_compile_options(so3lr_native_dev18_attention_scatter_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev19_block2_attention_input_reverse_test tests/test_kokkos_block2_attention_input_reverse.cpp)
  target_link_libraries(so3lr_native_dev19_block2_attention_input_reverse_test PRIVATE so3lr_block2_attention_input_reverse)
  target_compile_options(so3lr_native_dev19_block2_attention_input_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev20_block1_reverse_test tests/test_kokkos_complete_transformer_reverse.cpp)
  target_link_libraries(so3lr_native_dev20_block1_reverse_test PRIVATE so3lr_complete_transformer_reverse)
  target_compile_options(so3lr_native_dev20_block1_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev21_block0_reverse_test tests/test_kokkos_complete_transformer_block0_reverse.cpp)
  target_link_libraries(so3lr_native_dev21_block0_reverse_test PRIVATE so3lr_complete_transformer_reverse)
  target_compile_options(so3lr_native_dev21_block0_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev22_energy_reverse_chain_test tests/test_kokkos_learned_energy_reverse.cpp)
  target_link_libraries(so3lr_native_dev22_energy_reverse_chain_test PRIVATE so3lr_learned_energy_reverse)
  target_compile_options(so3lr_native_dev22_energy_reverse_chain_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev23_cartesian_force_test tests/test_kokkos_learned_energy_forces.cpp)
  target_link_libraries(so3lr_native_dev23_cartesian_force_test PRIVATE so3lr_learned_energy_forces)
  target_compile_options(so3lr_native_dev23_cartesian_force_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev24_multihead_force_test tests/test_kokkos_multihead_forces.cpp)
  target_link_libraries(so3lr_native_dev24_multihead_force_test PRIVATE so3lr_learned_energy_forces)
  target_compile_options(so3lr_native_dev24_multihead_force_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_physical_long_range STATIC src/kokkos_physical_long_range.cpp)
  target_include_directories(so3lr_physical_long_range PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_physical_long_range PUBLIC so3lr_learned_energy_forces)
  target_compile_options(so3lr_physical_long_range PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev25_physical_force_test tests/test_kokkos_physical_long_range.cpp)
  target_link_libraries(so3lr_native_dev25_physical_force_test PRIVATE so3lr_physical_long_range)
  target_compile_options(so3lr_native_dev25_physical_force_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_physical_zbl STATIC src/kokkos_physical_zbl.cpp
                                      src/so3lr_repulsion_setup.cpp)
  target_include_directories(so3lr_physical_zbl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_physical_zbl PUBLIC so3lr_learned_energy_forces)
  target_compile_options(so3lr_physical_zbl PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev56_zbl_test tests/test_kokkos_zbl.cpp)
  target_link_libraries(so3lr_native_dev56_zbl_test PRIVATE so3lr_physical_zbl)
  target_compile_options(so3lr_native_dev56_zbl_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_evaluator STATIC src/kokkos_so3lr_evaluator.cpp)
  target_include_directories(so3lr_evaluator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_evaluator PUBLIC so3lr_physical_long_range so3lr_physical_zbl)
  target_compile_options(so3lr_evaluator PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev26_evaluator_test tests/test_kokkos_so3lr_evaluator.cpp)
  target_link_libraries(so3lr_native_dev26_evaluator_test PRIVATE so3lr_evaluator)
  target_compile_options(so3lr_native_dev26_evaluator_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_ownership STATIC src/kokkos_so3lr_ownership.cpp)
  target_include_directories(so3lr_ownership PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_ownership PUBLIC so3lr_evaluator)
  target_compile_options(so3lr_ownership PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev27_ownership_test tests/test_kokkos_so3lr_ownership.cpp)
  target_link_libraries(so3lr_native_dev27_ownership_test PRIVATE so3lr_ownership)
  target_compile_options(so3lr_native_dev27_ownership_test PRIVATE -Wall -Wextra -Wpedantic)

  find_package(MPI REQUIRED COMPONENTS CXX)
  add_executable(so3lr_native_dev28_cuda_mpi_test tests/test_kokkos_so3lr_cuda_mpi.cpp)
  target_link_libraries(so3lr_native_dev28_cuda_mpi_test PRIVATE so3lr_ownership MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev28_cuda_mpi_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_feature_exchange STATIC src/kokkos_node_feature_exchange.cpp)
  target_include_directories(so3lr_feature_exchange PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_feature_exchange PUBLIC so3lr_transformer_stack012)
  target_compile_options(so3lr_feature_exchange PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev29_feature_exchange_test tests/test_kokkos_node_feature_exchange.cpp)
  target_link_libraries(so3lr_native_dev29_feature_exchange_test PRIVATE so3lr_feature_exchange MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev29_feature_exchange_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_local_transformer_forward STATIC src/kokkos_local_transformer_forward.cpp)
  target_include_directories(so3lr_local_transformer_forward PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_local_transformer_forward PUBLIC so3lr_feature_exchange)
  target_compile_options(so3lr_local_transformer_forward PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev30_local_forward_test tests/test_kokkos_local_transformer_forward.cpp)
  target_link_libraries(so3lr_native_dev30_local_forward_test PRIVATE so3lr_local_transformer_forward MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev30_local_forward_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_local_energy_reverse STATIC src/kokkos_local_energy_reverse.cpp)
  target_include_directories(so3lr_local_energy_reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_local_energy_reverse PUBLIC so3lr_complete_transformer_reverse so3lr_output_heads)
  target_compile_options(so3lr_local_energy_reverse PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev31_local_reverse_test tests/test_kokkos_local_energy_reverse.cpp)
  target_link_libraries(so3lr_native_dev31_local_reverse_test PRIVATE so3lr_local_energy_reverse so3lr_feature_exchange MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev31_local_reverse_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_local_cartesian_forces STATIC src/kokkos_local_cartesian_forces.cpp)
  target_include_directories(so3lr_local_cartesian_forces PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_local_cartesian_forces PUBLIC so3lr_local_energy_reverse so3lr_learned_energy_forces)
  target_compile_options(so3lr_local_cartesian_forces PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev32_local_cartesian_force_test tests/test_kokkos_local_cartesian_forces.cpp)
  target_link_libraries(so3lr_native_dev32_local_cartesian_force_test PRIVATE so3lr_local_cartesian_forces so3lr_feature_exchange MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev32_local_cartesian_force_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_rank_local_graph STATIC src/rank_local_graph.cpp)
  target_include_directories(so3lr_rank_local_graph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_rank_local_graph PUBLIC so3lr_native)
  target_compile_options(so3lr_rank_local_graph PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev33_rank_local_graph_test tests/test_kokkos_rank_local_graph.cpp)
  target_link_libraries(so3lr_native_dev33_rank_local_graph_test PRIVATE so3lr_rank_local_graph so3lr_local_cartesian_forces so3lr_feature_exchange MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev33_rank_local_graph_test PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev34_distributed_multihead_test tests/test_kokkos_distributed_multihead.cpp)
  target_link_libraries(so3lr_native_dev34_distributed_multihead_test PRIVATE so3lr_rank_local_graph so3lr_local_cartesian_forces so3lr_feature_exchange MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev34_distributed_multihead_test PRIVATE -Wall -Wextra -Wpedantic)


  add_library(so3lr_rank_local_half_pairs STATIC src/rank_local_half_pair_graph.cpp)
  target_include_directories(so3lr_rank_local_half_pairs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_rank_local_half_pairs PUBLIC so3lr_native)
  target_compile_options(so3lr_rank_local_half_pairs PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_distributed_physical_lr STATIC
    src/kokkos_physical_long_range_model.cpp
    src/kokkos_physical_long_range_exchange.cpp)
  target_include_directories(so3lr_distributed_physical_lr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_distributed_physical_lr PUBLIC so3lr_physical_long_range)
  target_compile_options(so3lr_distributed_physical_lr PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev35_distributed_physical_test tests/test_kokkos_distributed_physical.cpp)
  target_link_libraries(so3lr_native_dev35_distributed_physical_test PRIVATE
    so3lr_rank_local_graph so3lr_rank_local_half_pairs
    so3lr_local_cartesian_forces so3lr_feature_exchange
    so3lr_distributed_physical_lr MPI::MPI_CXX)
  target_compile_options(so3lr_native_dev35_distributed_physical_test PRIVATE -Wall -Wextra -Wpedantic)


  # Turbo mode: many independent single-rank images evaluated as one packed
  # batch on a GPU (pair_style so3lr/turbo). Built on the generalized
  # local-forces path, so it supports every model so3lr/native/mpi supports.
  add_library(so3lr_image_segments STATIC src/kokkos_image_segments.cpp)
  target_include_directories(so3lr_image_segments PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_image_segments PUBLIC Kokkos::kokkos)
  target_compile_options(so3lr_image_segments PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_turbo_image_segments_test tests/test_kokkos_image_segments.cpp)
  target_link_libraries(so3lr_turbo_image_segments_test PRIVATE so3lr_image_segments)
  target_compile_options(so3lr_turbo_image_segments_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_packed_physical STATIC src/kokkos_packed_so3lr_evaluator.cpp)
  target_include_directories(so3lr_packed_physical PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_packed_physical PUBLIC
    so3lr_local_cartesian_forces so3lr_distributed_physical_lr
    so3lr_physical_zbl so3lr_image_segments)
  target_compile_options(so3lr_packed_physical PRIVATE -Wall -Wextra -Wpedantic)

  # The packed evaluator test compares with the single-rank reference evaluator when
  # the model allows it, and otherwise with one-image-at-a-time packed runs.
  add_executable(so3lr_turbo_packed_physical_test tests/test_kokkos_packed_so3lr_evaluator.cpp)
  target_link_libraries(so3lr_turbo_packed_physical_test PRIVATE so3lr_packed_physical so3lr_evaluator)
  target_compile_options(so3lr_turbo_packed_physical_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_turbo_replica_bridge STATIC src/kokkos_turbo_replica_bridge.cpp)
  target_include_directories(so3lr_turbo_replica_bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_turbo_replica_bridge PUBLIC so3lr_packed_physical)
  target_compile_options(so3lr_turbo_replica_bridge PRIVATE -Wall -Wextra -Wpedantic)

  # Bridge and broker tests use the single-rank reference evaluator:
  # run them with an SO3LR model.
  add_executable(so3lr_turbo_replica_bridge_test tests/test_kokkos_turbo_replica_bridge.cpp)
  target_link_libraries(so3lr_turbo_replica_bridge_test PRIVATE so3lr_turbo_replica_bridge so3lr_evaluator)
  target_compile_options(so3lr_turbo_replica_bridge_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_mpi_turbo_replica_broker STATIC src/mpi_turbo_replica_broker.cpp)
  target_include_directories(so3lr_mpi_turbo_replica_broker PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_mpi_turbo_replica_broker PUBLIC so3lr_turbo_replica_bridge MPI::MPI_CXX)
  target_compile_options(so3lr_mpi_turbo_replica_broker PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_turbo_mpi_replica_broker_test tests/test_mpi_turbo_replica_broker.cpp)
  target_link_libraries(so3lr_turbo_mpi_replica_broker_test PRIVATE
    so3lr_mpi_turbo_replica_broker so3lr_evaluator MPI::MPI_CXX)
  target_compile_options(so3lr_turbo_mpi_replica_broker_test PRIVATE -Wall -Wextra -Wpedantic)

  add_library(so3lr_lammps_neighbor_adapter STATIC src/lammps_neighbor_adapter.cpp)
  target_include_directories(so3lr_lammps_neighbor_adapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(so3lr_lammps_neighbor_adapter PUBLIC so3lr_native)
  target_compile_options(so3lr_lammps_neighbor_adapter PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(so3lr_native_dev36_lammps_adapter_test tests/test_lammps_neighbor_adapter.cpp)
  target_link_libraries(so3lr_native_dev36_lammps_adapter_test PRIVATE so3lr_lammps_neighbor_adapter)
  target_compile_options(so3lr_native_dev36_lammps_adapter_test PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
# Needs a generated fixture; registering it unconditionally made a CPU-only
# configure report a failure for a test it was never given inputs for.
if(SO3LR_TEST_MODEL AND SO3LR_INPUT_FIXTURE)
  add_test(NAME native_loader_and_inputs COMMAND so3lr_native_dev2_test ${SO3LR_TEST_MODEL} ${SO3LR_INPUT_FIXTURE})
endif()
if(SO3LR_ENABLE_KOKKOS)
  add_test(NAME kokkos_cuda_input_operators COMMAND so3lr_native_dev3_gpu_test ${SO3LR_TEST_MODEL} ${SO3LR_INPUT_FIXTURE})
  add_test(NAME kokkos_cuda_block0_filters COMMAND so3lr_native_dev4_filter_test ${SO3LR_TEST_MODEL} ${SO3LR_FILTER_FIXTURE})
  add_test(NAME cublas_block0_filter_ab COMMAND so3lr_native_dev5_blas_test ${SO3LR_TEST_MODEL} ${SO3LR_FILTER_FIXTURE})
  add_test(NAME cublas_block0_qkv COMMAND so3lr_native_dev6_qkv_test ${SO3LR_TEST_MODEL} ${SO3LR_QKV_FIXTURE})
  add_test(NAME fused_block0_attention_scatter COMMAND so3lr_native_dev7_attention_test ${SO3LR_TEST_MODEL} ${SO3LR_ATTENTION_FIXTURE})
  add_test(NAME integrated_block0_attention_pipeline COMMAND so3lr_native_dev8_pipeline_test ${SO3LR_TEST_MODEL} ${SO3LR_INTEGRATED_FIXTURE})
  add_test(NAME post_attention_block0_pipeline COMMAND so3lr_native_dev9_post_attention_test ${SO3LR_TEST_MODEL} ${SO3LR_POST_ATTENTION_FIXTURE})
  add_test(NAME complete_transformer_block0 COMMAND so3lr_native_dev10_transformer_block0_test ${SO3LR_TEST_MODEL} ${SO3LR_FULL_BLOCK0_FIXTURE})
  add_test(NAME feature_input_transformer_block1 COMMAND so3lr_native_dev11_feature_transformer_test ${SO3LR_TEST_MODEL} ${SO3LR_BLOCK1_FIXTURE})
  add_test(NAME transformer_stack_0_1 COMMAND so3lr_native_dev12_transformer_stack01_test ${SO3LR_TEST_MODEL} ${SO3LR_STACK01_FIXTURE})
  add_test(NAME complete_transformer_stack_0_1_2 COMMAND so3lr_native_dev13_transformer_stack012_test ${SO3LR_TEST_MODEL} ${SO3LR_STACK012_FIXTURE})
  add_test(NAME native_energy_charge_hirshfeld_heads COMMAND so3lr_native_dev14_output_heads_test ${SO3LR_TEST_MODEL} ${SO3LR_OUTPUT_HEADS_FIXTURE})
  add_test(NAME complete_native_learned_forward COMMAND so3lr_native_dev15_forward_test ${SO3LR_TEST_MODEL} ${SO3LR_FORWARD_FIXTURE})
  add_test(NAME native_energy_head_reverse COMMAND so3lr_native_dev16_energy_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_ENERGY_REVERSE_FIXTURE})
  add_test(NAME native_block2_post_attention_reverse COMMAND so3lr_native_dev17_post_attention_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_POST_ATTENTION_REVERSE_FIXTURE})
  add_test(NAME native_block2_attention_scatter_reverse COMMAND so3lr_native_dev18_attention_scatter_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_ATTENTION_SCATTER_REVERSE_FIXTURE})
  add_test(NAME native_block2_attention_input_reverse COMMAND so3lr_native_dev19_block2_attention_input_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_BLOCK2_ATTENTION_INPUT_REVERSE_FIXTURE})
  add_test(NAME native_complete_block1_reverse COMMAND so3lr_native_dev20_block1_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_BLOCK1_REVERSE_FIXTURE})
  add_test(NAME native_complete_block0_reverse COMMAND so3lr_native_dev21_block0_reverse_test ${SO3LR_TEST_MODEL} ${SO3LR_BLOCK0_REVERSE_FIXTURE})
  add_test(NAME native_learned_energy_reverse_chain COMMAND so3lr_native_dev22_energy_reverse_chain_test ${SO3LR_TEST_MODEL} ${SO3LR_ENERGY_REVERSE_CHAIN_FIXTURE})
  add_test(NAME native_learned_energy_cartesian_forces COMMAND so3lr_native_dev23_cartesian_force_test ${SO3LR_TEST_MODEL} ${SO3LR_CARTESIAN_FORCE_FIXTURE})
  add_test(NAME native_energy_charge_hirshfeld_cartesian_forces COMMAND so3lr_native_dev24_multihead_force_test ${SO3LR_TEST_MODEL} ${SO3LR_MULTIHEAD_FORCE_FIXTURE})
  add_test(NAME native_physical_long_range_cartesian_forces COMMAND so3lr_native_dev25_physical_force_test ${SO3LR_TEST_MODEL} ${SO3LR_PHYSICAL_FORCE_FIXTURE})
  add_test(NAME native_self_contained_so3lr_evaluator COMMAND so3lr_native_dev26_evaluator_test ${SO3LR_TEST_MODEL} ${SO3LR_PHYSICAL_FORCE_FIXTURE})
  add_test(NAME native_owned_ghost_force_contract COMMAND so3lr_native_dev27_ownership_test ${SO3LR_TEST_MODEL} ${SO3LR_PHYSICAL_FORCE_FIXTURE})
endif()
