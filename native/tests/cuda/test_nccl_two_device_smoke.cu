#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__global__ void ScaleKernel(float* values, int count, float scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        values[idx] *= scale;
    }
}

int CheckCuda(cudaError_t status) {
    return status == cudaSuccess ? 0 : 1;
}

int CheckNccl(ncclResult_t status) {
    return status == ncclSuccess ? 0 : 1;
}

}  // namespace

int main() {
    int device_count = 0;
    if (CheckCuda(cudaGetDeviceCount(&device_count)) != 0) return EXIT_FAILURE;
    if (device_count < 2) {
        std::puts("skip: fewer than two CUDA devices visible");
        return EXIT_SUCCESS;
    }

    constexpr int kRanks = 2;
    constexpr int kCount = 16;
    const int devices[kRanks] = {0, 1};
    ncclComm_t comms[kRanks]{};
    if (CheckNccl(ncclCommInitAll(comms, kRanks, devices)) != 0) return EXIT_FAILURE;

    float* buffers[kRanks] = {nullptr, nullptr};
    cudaStream_t streams[kRanks]{};
    for (int rank = 0; rank < kRanks; ++rank) {
        if (CheckCuda(cudaSetDevice(devices[rank])) != 0) return EXIT_FAILURE;
        if (CheckCuda(cudaStreamCreate(&streams[rank])) != 0) return EXIT_FAILURE;
        if (CheckCuda(cudaMalloc(&buffers[rank], kCount * sizeof(float))) != 0) return EXIT_FAILURE;
        std::vector<float> host(kCount, static_cast<float>(rank + 1));
        if (CheckCuda(cudaMemcpy(buffers[rank], host.data(), kCount * sizeof(float), cudaMemcpyHostToDevice)) != 0) return EXIT_FAILURE;
    }

    ncclGroupStart();
    for (int rank = 0; rank < kRanks; ++rank) {
        ncclAllReduce(buffers[rank], buffers[rank], kCount, ncclFloat32, ncclSum, comms[rank], streams[rank]);
    }
    if (CheckNccl(ncclGroupEnd()) != 0) return EXIT_FAILURE;

    for (int rank = 0; rank < kRanks; ++rank) {
        if (CheckCuda(cudaSetDevice(devices[rank])) != 0) return EXIT_FAILURE;
        ScaleKernel<<<1, 32, 0, streams[rank]>>>(buffers[rank], kCount, 1.0f / static_cast<float>(kRanks));
        if (CheckCuda(cudaStreamSynchronize(streams[rank])) != 0) return EXIT_FAILURE;
        std::vector<float> host(kCount);
        if (CheckCuda(cudaMemcpy(host.data(), buffers[rank], kCount * sizeof(float), cudaMemcpyDeviceToHost)) != 0) return EXIT_FAILURE;
        for (float value : host) {
            if (std::fabs(value - 1.5f) > 1.0e-6f) return EXIT_FAILURE;
        }
    }

    for (int rank = 0; rank < kRanks; ++rank) {
        cudaSetDevice(devices[rank]);
        cudaFree(buffers[rank]);
        cudaStreamDestroy(streams[rank]);
        if (CheckNccl(ncclCommDestroy(comms[rank])) != 0) return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}