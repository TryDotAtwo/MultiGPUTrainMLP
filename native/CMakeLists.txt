cmake_minimum_required(VERSION 3.22)
project(multigpu_train_mlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MGT_ENABLE_CUDA "Build CUDA trainer code" ON)
option(MGT_ENABLE_NCCL "Build NCCL allreduce code" ON)

if(MGT_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(mgt_core STATIC
    src/static_contracts.cpp
    src/puzzle_io.cpp
    src/model_layout.cpp
    src/weight_export.cpp
    src/random_walk_cpu.cpp
    src/mlp_cpu_ref.cpp
    src/memory_plan.cpp
    src/allreduce.cpp
)

target_include_directories(mgt_core PUBLIC include)

add_executable(test_static_contracts tests/test_static_contracts.cpp)
target_link_libraries(test_static_contracts PRIVATE mgt_core)

enable_testing()
add_test(NAME static_contracts COMMAND test_static_contracts)
add_executable(test_puzzle_io tests/test_puzzle_io.cpp)
target_link_libraries(test_puzzle_io PRIVATE mgt_core)
add_test(NAME puzzle_io COMMAND test_puzzle_io)
set_tests_properties(puzzle_io PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/..")
add_executable(test_model_layout tests/test_model_layout.cpp)
target_link_libraries(test_model_layout PRIVATE mgt_core)
add_test(NAME model_layout COMMAND test_model_layout)

add_executable(test_weight_export tests/test_weight_export.cpp)
target_link_libraries(test_weight_export PRIVATE mgt_core)
add_test(NAME weight_export COMMAND test_weight_export)
set_tests_properties(weight_export PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/..")
add_executable(test_random_walk_cpu tests/test_random_walk_cpu.cpp)
target_link_libraries(test_random_walk_cpu PRIVATE mgt_core)
add_test(NAME random_walk_cpu COMMAND test_random_walk_cpu)
add_executable(test_mlp_cpu_ref tests/test_mlp_cpu_ref.cpp)
target_link_libraries(test_mlp_cpu_ref PRIVATE mgt_core)
add_test(NAME mlp_cpu_ref COMMAND test_mlp_cpu_ref)
add_executable(test_memory_plan tests/test_memory_plan.cpp)
target_link_libraries(test_memory_plan PRIVATE mgt_core)
add_test(NAME memory_plan COMMAND test_memory_plan)
add_executable(test_allreduce tests/test_allreduce.cpp)
target_link_libraries(test_allreduce PRIVATE mgt_core)
add_test(NAME allreduce COMMAND test_allreduce)
if(MGT_ENABLE_CUDA)
    add_library(mgt_cuda STATIC
        cuda/device_context.cu
        cuda/random_walk_kernel.cu
        cuda/mlp_forward.cu
        cuda/mlp_backward.cu
        cuda/adamw.cu
    )
    target_include_directories(mgt_cuda PUBLIC include cuda)
    target_link_libraries(mgt_cuda PUBLIC mgt_core CUDA::cublas)
    find_path(NVTX3_INCLUDE_DIR nvtx3/nvToolsExt.h PATHS
        "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include"
        "${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/targets/x86_64-linux/include")
    if(NVTX3_INCLUDE_DIR)
        target_include_directories(mgt_cuda PUBLIC ${NVTX3_INCLUDE_DIR})
        target_compile_definitions(mgt_cuda PUBLIC MGT_HAS_NVTX=1)
    endif()

    if(MGT_ENABLE_NCCL)
        find_library(NCCL_LIB nccl)
        find_path(NCCL_INCLUDE_DIR nccl.h)
        if(NCCL_LIB AND NCCL_INCLUDE_DIR)
            target_sources(mgt_cuda PRIVATE cuda/allreduce_nccl.cu)
            target_include_directories(mgt_cuda PUBLIC ${NCCL_INCLUDE_DIR})
            target_link_libraries(mgt_cuda PUBLIC ${NCCL_LIB})
            target_compile_definitions(mgt_cuda PUBLIC MGT_HAS_NCCL=1)
            add_executable(test_nccl_single_rank_smoke tests/cuda/test_nccl_single_rank_smoke.cu)
            target_link_libraries(test_nccl_single_rank_smoke PRIVATE mgt_cuda)
            add_test(NAME nccl_single_rank_smoke COMMAND test_nccl_single_rank_smoke)
            add_executable(test_nccl_two_device_smoke tests/cuda/test_nccl_two_device_smoke.cu)
            target_link_libraries(test_nccl_two_device_smoke PRIVATE mgt_cuda)
            add_test(NAME nccl_two_device_smoke COMMAND test_nccl_two_device_smoke)
        else()
            message(STATUS "NCCL not found; skipping NCCL tests")
        endif()
    endif()
    add_executable(test_cuda_compile tests/cuda/test_cuda_compile.cu)
    target_link_libraries(test_cuda_compile PRIVATE mgt_cuda)
    add_test(NAME cuda_compile COMMAND test_cuda_compile)
    add_executable(test_cuda_random_walk_smoke tests/cuda/test_cuda_random_walk_smoke.cu)
    target_link_libraries(test_cuda_random_walk_smoke PRIVATE mgt_cuda)
    add_test(NAME cuda_random_walk_smoke COMMAND test_cuda_random_walk_smoke)
    add_executable(test_cuda_adamw_smoke tests/cuda/test_cuda_adamw_smoke.cu)
    target_link_libraries(test_cuda_adamw_smoke PRIVATE mgt_cuda)
    add_test(NAME cuda_adamw_smoke COMMAND test_cuda_adamw_smoke)
    add_executable(test_cuda_mlp_forward_smoke tests/cuda/test_cuda_mlp_forward_smoke.cu)
    target_link_libraries(test_cuda_mlp_forward_smoke PRIVATE mgt_cuda)
    add_test(NAME cuda_mlp_forward_smoke COMMAND test_cuda_mlp_forward_smoke)
    add_executable(test_cuda_mlp_backward_smoke tests/cuda/test_cuda_mlp_backward_smoke.cu)
    target_link_libraries(test_cuda_mlp_backward_smoke PRIVATE mgt_cuda)
    add_test(NAME cuda_mlp_backward_smoke COMMAND test_cuda_mlp_backward_smoke)
    add_executable(test_cuda_mlp_backward_cpu_compare tests/cuda/test_cuda_mlp_backward_cpu_compare.cpp)
    target_link_libraries(test_cuda_mlp_backward_cpu_compare PRIVATE mgt_cuda)
    add_test(NAME cuda_mlp_backward_cpu_compare COMMAND test_cuda_mlp_backward_cpu_compare)
    add_executable(test_cuda_train_step_smoke tests/cuda/test_cuda_train_step_smoke.cu)
    target_link_libraries(test_cuda_train_step_smoke PRIVATE mgt_cuda)
    add_test(NAME cuda_train_step_smoke COMMAND test_cuda_train_step_smoke)
    add_executable(mgt_native_train tools/mgt_native_train_smoke.cu)
    target_link_libraries(mgt_native_train PRIVATE mgt_cuda)
    add_executable(mgt_native_train_smoke tools/mgt_native_train_smoke.cu)
    target_link_libraries(mgt_native_train_smoke PRIVATE mgt_cuda)
    add_test(NAME native_train_smoke COMMAND mgt_native_train_smoke --output-dir native-train-smoke-run --steps 2 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0)
    if(MGT_ENABLE_NCCL AND NCCL_LIB AND NCCL_INCLUDE_DIR)
        add_test(NAME native_train_nccl_single_rank_smoke COMMAND mgt_native_train_smoke --output-dir native-train-nccl-smoke-run --steps 2 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --nccl-id-file native-train-nccl-smoke-run/nccl.id)
    endif()
    add_test(NAME native_train_profile_smoke COMMAND mgt_native_train_smoke --output-dir native-train-profile-smoke-run --steps 1 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --batch-size 17 --k-min 2 --k-max 5 --hd1 7 --hd2 4)
    add_test(NAME native_train_resume_smoke COMMAND mgt_native_train_smoke --output-dir native-train-resume-smoke-run --steps 1 --device-id 0 --world-size 1 --global-rank 0 --local-rank 0 --resume-checkpoint native-train-smoke-run/checkpoint)
    set_tests_properties(native_train_resume_smoke PROPERTIES DEPENDS native_train_smoke)
    add_test(NAME native_train_artifacts COMMAND bash ${CMAKE_SOURCE_DIR}/tests/scripts/check_native_train_artifacts.sh native-train-smoke-run)
    set_tests_properties(native_train_artifacts PROPERTIES DEPENDS native_train_smoke)
endif()
