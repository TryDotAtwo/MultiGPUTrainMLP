#include "mgt/config.hpp"
#include "mgt/single_gpu_contract.hpp"

#include <cstdlib>

int main() {
    using Legacy = mgt::P888TrainingContract;
    if (Legacy::kInputFeatures != 5184) return EXIT_FAILURE;

    const auto contract = mgt::OriginalP888SingleGpuContract();
    if (contract.schema_version != 1) return EXIT_FAILURE;
    if (contract.state_len != 72 || contract.state_value_count != 72) return EXIT_FAILURE;
    if (contract.input_features != 5184) return EXIT_FAILURE;
    if (contract.logical_hd1 != 2556 || contract.physical_hd1 != 2560) return EXIT_FAILURE;
    if (contract.logical_hd2 != 218 || contract.physical_hd2 != 224) return EXIT_FAILURE;
    if (contract.residual_blocks != 16 || contract.batch_norm_sites != 34) return EXIT_FAILURE;
    if (contract.output_dim != 1) return EXIT_FAILURE;
    if (mgt::ValidateSingleGpuModelContract(contract) != mgt::Status::kOk) return EXIT_FAILURE;

    auto bad = contract;
    bad.input_features = 6336;
    if (mgt::ValidateSingleGpuModelContract(bad) != mgt::Status::kInvalidConfig) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
