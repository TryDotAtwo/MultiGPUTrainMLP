#pragma once

#include "mgt/model_layout.hpp"
#include <filesystem>
#include <span>

namespace mgt {

Status ExportInferenceWeights(const std::filesystem::path& output_dir,
                              const ModelLayout& layout,
                              std::span<const float> weights);

}  // namespace mgt