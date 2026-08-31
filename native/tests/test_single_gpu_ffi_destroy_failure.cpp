// Link the real C ABI adapter to a test-only backend. No GPU/context failure
// injection is required to exercise ownership after a failed native teardown.
#include "../src/single_gpu_trainer_ffi.cpp"

#include <cstdio>
#include <stdexcept>

namespace mgt_cuda {
struct SingleGpuTrainer {};
static mgt::Status destroy_status = mgt::Status::kOk;
static unsigned destroy_calls = 0;

mgt::Status CreateSingleGpuTrainer(const SingleGpuTrainerCreateInfo&, SingleGpuTrainer**) {
    return mgt::Status::kInvalidConfig;
}
mgt::Status PrepareSingleGpuTrainer(SingleGpuTrainer*) { return mgt::Status::kInvalidConfig; }
mgt::Status LaunchSingleGpuTrainStep(SingleGpuTrainer*, const SingleGpuTrainStepRequest&,
                                   SingleGpuTrainStepTicket*) { return mgt::Status::kInvalidConfig; }
mgt::Status ReadSingleGpuMetrics(SingleGpuTrainer*, SingleGpuTrainerMetrics*) {
    return mgt::Status::kInvalidConfig;
}
mgt::Status DestroySingleGpuTrainer(SingleGpuTrainer** pointer) {
    ++destroy_calls;
    if (!pointer || !*pointer) return mgt::Status::kInvalidConfig;
    delete *pointer;
    *pointer = nullptr;
    return destroy_status;
}
}  // namespace mgt_cuda

int main() {
    try {
        for (auto result : {mgt::Status::kOk, mgt::Status::kCudaFailure}) {
            auto* handle = new MgtSingleGpuHandle;
            handle->trainer = new mgt_cuda::SingleGpuTrainer;
            mgt_cuda::destroy_status = result;
            const auto status = mgt_single_gpu_v1_destroy(&handle);
            const bool consumed = handle == nullptr;
            // Keep the deliberately failing RED test leak-free too.
            if (handle) { delete handle->trainer; delete handle; }
            if (status != Convert(result) || !consumed)
                throw std::runtime_error("destroy must consume handle even on CUDA failure");
            if (result != mgt::Status::kOk) {
                char error[128]{};
                mgt_single_gpu_v1_last_error(nullptr, error, sizeof(error));
                if (std::string(error) != "native trainer destroy failed")
                    throw std::runtime_error("destroy failure lost its thread-local error");
            }
        }
        MgtSingleGpuHandle* empty = nullptr;
        if (mgt_single_gpu_v1_destroy(&empty) != MGT_STATUS_INVALID_CONFIG ||
            mgt_single_gpu_v1_destroy(nullptr) != MGT_STATUS_INVALID_CONFIG ||
            mgt_cuda::destroy_calls != 2)
            throw std::runtime_error("invalid destroy called backend");
        std::puts("PASS C ABI consumes handle on failed native teardown");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL destroy ownership: %s\n", error.what());
        return 1;
    }
}
