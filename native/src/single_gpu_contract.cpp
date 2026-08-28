#include "mgt/single_gpu_contract.hpp"

#include "mgt/config.hpp"

#include <cstdint>
#include <limits>

namespace mgt {

SingleGpuModelContract OriginalP888SingleGpuContract() {
    return SingleGpuModelContract{
        1,
        kStateLen,
        P888TrainingContract::kStateValueCount,
        P888TrainingContract::kInputFeatures,
        P888TrainingContract::kHidden1,
        2560,
        P888TrainingContract::kHidden2,
        224,
        P888TrainingContract::kResidualBlocks,
        P888TrainingContract::kBatchNormSites,
        1,
    };
}

Status ValidateSingleGpuModelContract(const SingleGpuModelContract& contract) {
    const std::uint64_t encoded_features =
        static_cast<std::uint64_t>(contract.state_len) * contract.state_value_count;
    if (contract.schema_version != 1 || contract.state_len == 0 ||
        contract.state_value_count == 0 ||
        encoded_features > std::numeric_limits<std::uint32_t>::max() ||
        contract.input_features != encoded_features ||
        contract.logical_hd1 != P888TrainingContract::kHidden1 ||
        contract.logical_hd2 != P888TrainingContract::kHidden2 ||
        contract.physical_hd1 < contract.logical_hd1 ||
        contract.physical_hd2 < contract.logical_hd2 ||
        contract.physical_hd1 % 8 != 0 || contract.physical_hd2 % 8 != 0 ||
        contract.residual_blocks != P888TrainingContract::kResidualBlocks ||
        contract.batch_norm_sites != P888TrainingContract::kBatchNormSites ||
        contract.output_dim != 1) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

}  // namespace mgt
