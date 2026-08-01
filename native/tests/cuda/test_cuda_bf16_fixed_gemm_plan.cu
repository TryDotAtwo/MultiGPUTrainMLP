#include "mgt_cuda/bf16_linear_train_ops.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    mgt_cuda::FixedBf16GemmPlan* plan = nullptr;
    mgt::Bf16GemmKeyV1 key{};
    mgt::Bf16GemmChoiceV1 choice{};
    if (mgt_cuda::CreateFixedBf16GemmPlan(nullptr, key, choice, nullptr, 0, &plan) ==
            mgt::Status::kOk || plan != nullptr) return 1;
    if (mgt_cuda::DestroyFixedBf16GemmPlan(nullptr) == mgt::Status::kOk) return 2;
    if (mgt_cuda::LaunchFixedBf16Gemm(nullptr, nullptr, nullptr, nullptr, nullptr) ==
        mgt::Status::kOk) return 3;
    return 0;
}
