#pragma once
#include "mgt/static_contracts.hpp"
#include <cstdint>
#include <vector>
namespace mgt {
struct BatchNormWorkspaceSlice {std::uint32_t logical_features=0,physical_stride=0;std::uint64_t affine_offset=0,running_offset=0,normalized_offset=0,mean_offset=0,inv_std_offset=0;};
struct BatchNormTrainingPlan {std::vector<BatchNormWorkspaceSlice> sites;std::uint64_t logical_feature_count=0,storage_feature_count=0,trainable_count=0,running_count=0,normalized_count=0,mean_offset=0,inv_std_offset=0,reduction_offset=0,reduction_count=0,workspace_floats=0;};
struct BatchNormTrainingState {std::vector<float> affine,running,adam_m,adam_v;};
Status BuildBatchNormTrainingPlan(std::uint32_t logical_hd1,std::uint32_t logical_hd2,std::uint32_t physical_hd1,std::uint32_t physical_hd2,std::uint32_t residual_blocks,std::uint32_t local_rows,BatchNormTrainingPlan* plan);
BatchNormTrainingState InitializeBatchNormTrainingState(const BatchNormTrainingPlan& plan);
}