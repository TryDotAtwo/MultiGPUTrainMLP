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
)

target_include_directories(mgt_core PUBLIC include)

add_executable(test_static_contracts tests/test_static_contracts.cpp)
target_link_libraries(test_static_contracts PRIVATE mgt_core)

enable_testing()
add_test(NAME static_contracts COMMAND test_static_contracts)