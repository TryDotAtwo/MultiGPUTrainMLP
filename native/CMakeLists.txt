cmake_minimum_required(VERSION 3.22)
project(multigpu_train_mlp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MGT_ENABLE_CUDA "Build CUDA trainer code" ON)
option(MGT_ENABLE_NCCL "Build NCCL allreduce code" ON)

if(MGT_ENABLE_CUDA)
    enable_language(CUDA)
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