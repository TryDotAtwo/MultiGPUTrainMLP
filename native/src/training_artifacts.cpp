#include "mgt/training_artifacts.hpp"

#include <array>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <string_view>

namespace mgt {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t UpdateFnv(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
    return hash;
}

bool ValidMetadata(const CheckpointMetadata& metadata) {
    return metadata.version == 2 && metadata.param_count > 0 &&
           metadata.payload_bytes > 0 && !metadata.fingerprint.empty() &&
           metadata.learning_rate > 0.0f && metadata.weight_decay >= 0.0f &&
           metadata.adam_beta1 >= 0.0f && metadata.adam_beta1 < 1.0f &&
           metadata.adam_beta2 >= 0.0f && metadata.adam_beta2 < 1.0f &&
           metadata.adam_eps > 0.0f;
}

template <typename T>
bool ParseInteger(const std::string& text, T* value) {
    if (value == nullptr || text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto result = std::from_chars(begin, end, *value);
    return result.ec == std::errc{} && result.ptr == end;
}

bool ParseFloat(const std::string& text, float* value) {
    if (value == nullptr || text.empty()) return false;
    try {
        std::size_t used = 0;
        const float parsed = std::stof(text, &used);
        if (used != text.size()) return false;
        *value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool SameOptimizer(const CheckpointMetadata& lhs, const CheckpointMetadata& rhs) {
    return lhs.seed == rhs.seed &&
           lhs.learning_rate == rhs.learning_rate &&
           lhs.weight_decay == rhs.weight_decay &&
           lhs.adam_beta1 == rhs.adam_beta1 &&
           lhs.adam_beta2 == rhs.adam_beta2 &&
           lhs.adam_eps == rhs.adam_eps;
}

}  // namespace

std::uint64_t Fnv1a64(const void* data, std::size_t size) {
    if (data == nullptr && size != 0) return 0;
    return UpdateFnv(kFnvOffset, data, size);
}

Status WriteCheckpointMetadata(const std::filesystem::path& path,
                               const CheckpointMetadata& metadata) {
    if (!ValidMetadata(metadata)) return Status::kInvalidConfig;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return Status::kIoFailure;
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "format=mgt_train_checkpoint\n"
        << "version=" << metadata.version << "\n"
        << "completed_steps=" << metadata.completed_steps << "\n"
        << "param_count=" << metadata.param_count << "\n"
        << "payload_bytes=" << metadata.payload_bytes << "\n"
        << "payload_checksum=" << metadata.payload_checksum << "\n"
        << "fingerprint=" << metadata.fingerprint << "\n"
        << "seed=" << metadata.seed << "\n"
        << "learning_rate=" << metadata.learning_rate << "\n"
        << "weight_decay=" << metadata.weight_decay << "\n"
        << "adam_beta1=" << metadata.adam_beta1 << "\n"
        << "adam_beta2=" << metadata.adam_beta2 << "\n"
        << "adam_eps=" << metadata.adam_eps << "\n";
    out.close();
    return out ? Status::kOk : Status::kIoFailure;
}

Status ReadCheckpointMetadata(const std::filesystem::path& path,
                              CheckpointMetadata* metadata) {
    if (metadata == nullptr) return Status::kInvalidConfig;
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::kIoFailure;

    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos || equals == 0 ||
            !fields.emplace(line.substr(0, equals), line.substr(equals + 1)).second) {
            return Status::kInvalidConfig;
        }
    }
    if (!in.eof()) return Status::kIoFailure;

    constexpr std::array<std::string_view, 13> required{
        "format", "version", "completed_steps", "param_count", "payload_bytes",
        "payload_checksum", "fingerprint", "seed", "learning_rate", "weight_decay",
        "adam_beta1", "adam_beta2", "adam_eps"};
    if (fields.size() != required.size()) return Status::kInvalidConfig;
    for (const std::string_view key : required) {
        if (!fields.contains(std::string(key))) return Status::kInvalidConfig;
    }
    if (fields["format"] != "mgt_train_checkpoint") return Status::kInvalidConfig;

    CheckpointMetadata parsed{};
    if (!ParseInteger(fields["version"], &parsed.version) ||
        !ParseInteger(fields["completed_steps"], &parsed.completed_steps) ||
        !ParseInteger(fields["param_count"], &parsed.param_count) ||
        !ParseInteger(fields["payload_bytes"], &parsed.payload_bytes) ||
        !ParseInteger(fields["payload_checksum"], &parsed.payload_checksum) ||
        !ParseInteger(fields["seed"], &parsed.seed) ||
        !ParseFloat(fields["learning_rate"], &parsed.learning_rate) ||
        !ParseFloat(fields["weight_decay"], &parsed.weight_decay) ||
        !ParseFloat(fields["adam_beta1"], &parsed.adam_beta1) ||
        !ParseFloat(fields["adam_beta2"], &parsed.adam_beta2) ||
        !ParseFloat(fields["adam_eps"], &parsed.adam_eps)) {
        return Status::kInvalidConfig;
    }
    parsed.fingerprint = fields["fingerprint"];
    if (!ValidMetadata(parsed)) return Status::kInvalidConfig;
    *metadata = std::move(parsed);
    return Status::kOk;
}

Status ValidateCheckpointMetadata(const CheckpointMetadata& actual,
                                  const CheckpointMetadata& expected) {
    if (!ValidMetadata(actual) || !ValidMetadata(expected)) return Status::kInvalidConfig;
    if (actual.version != expected.version ||
        actual.param_count != expected.param_count ||
        actual.payload_bytes != expected.payload_bytes ||
        actual.payload_checksum != expected.payload_checksum ||
        actual.fingerprint != expected.fingerprint ||
        !SameOptimizer(actual, expected)) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

Status ValidateCheckpointCompatibility(const CheckpointMetadata& actual,
                                       const CheckpointMetadata& expected) {
    if (!ValidMetadata(actual) || !ValidMetadata(expected)) {
        return Status::kInvalidConfig;
    }
    if (actual.version != expected.version ||
        actual.param_count != expected.param_count ||
        actual.payload_bytes != expected.payload_bytes ||
        actual.fingerprint != expected.fingerprint ||
        !SameOptimizer(actual, expected)) {
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

std::uint64_t GlobalStep(std::uint64_t completed_steps,
                         std::uint64_t local_step) {
    return completed_steps + local_step;
}

Status WriteCheckpointPayload(const std::filesystem::path& path,
                              const std::vector<float>& weights,
                              const std::vector<float>& adam_m,
                              const std::vector<float>& adam_v,
                              CheckpointMetadata* metadata) {
    if (metadata == nullptr || weights.empty() ||
        weights.size() != adam_m.size() || weights.size() != adam_v.size() ||
        metadata->param_count != weights.size()) {
        return Status::kInvalidConfig;
    }
    const std::size_t section_bytes = weights.size() * sizeof(float);
    std::uint64_t checksum = kFnvOffset;
    checksum = UpdateFnv(checksum, weights.data(), section_bytes);
    checksum = UpdateFnv(checksum, adam_m.data(), section_bytes);
    checksum = UpdateFnv(checksum, adam_v.data(), section_bytes);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return Status::kIoFailure;
    out.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(section_bytes));
    out.write(reinterpret_cast<const char*>(adam_m.data()),
              static_cast<std::streamsize>(section_bytes));
    out.write(reinterpret_cast<const char*>(adam_v.data()),
              static_cast<std::streamsize>(section_bytes));
    out.close();
    if (!out) return Status::kIoFailure;
    metadata->payload_bytes = 3ULL * section_bytes;
    metadata->payload_checksum = checksum;
    return Status::kOk;
}

Status ReadCheckpointPayload(const std::filesystem::path& path,
                             const CheckpointMetadata& metadata,
                             std::vector<float>* weights,
                             std::vector<float>* adam_m,
                             std::vector<float>* adam_v) {
    if (weights == nullptr || adam_m == nullptr || adam_v == nullptr ||
        weights->size() != metadata.param_count ||
        adam_m->size() != metadata.param_count ||
        adam_v->size() != metadata.param_count) {
        return Status::kInvalidConfig;
    }
    const std::uint64_t expected_bytes =
        3ULL * metadata.param_count * sizeof(float);
    if (metadata.payload_bytes != expected_bytes) return Status::kInvalidConfig;
    std::error_code ec;
    const std::uint64_t actual_bytes = std::filesystem::file_size(path, ec);
    if (ec) return Status::kIoFailure;
    if (actual_bytes != expected_bytes) return Status::kInvalidConfig;

    const std::size_t section_bytes = weights->size() * sizeof(float);
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status::kIoFailure;
    in.read(reinterpret_cast<char*>(weights->data()),
            static_cast<std::streamsize>(section_bytes));
    in.read(reinterpret_cast<char*>(adam_m->data()),
            static_cast<std::streamsize>(section_bytes));
    in.read(reinterpret_cast<char*>(adam_v->data()),
            static_cast<std::streamsize>(section_bytes));
    if (!in) return Status::kIoFailure;
    std::uint64_t checksum = kFnvOffset;
    checksum = UpdateFnv(checksum, weights->data(), section_bytes);
    checksum = UpdateFnv(checksum, adam_m->data(), section_bytes);
    checksum = UpdateFnv(checksum, adam_v->data(), section_bytes);
    return checksum == metadata.payload_checksum
        ? Status::kOk : Status::kInvalidConfig;
}

bool ShouldWritePeriodicArtifact(std::uint64_t completed_steps,
                                 std::uint64_t period_steps,
                                 bool final_step) {
    if (final_step) return true;
    return period_steps != 0 && completed_steps != 0 &&
           completed_steps % period_steps == 0;
}

Status PublishDirectoryAtomically(const std::filesystem::path& staged,
                                  const std::filesystem::path& current) {
    if (staged.empty() || current.empty() || staged == current ||
        staged.parent_path() != current.parent_path() ||
        staged.filename() != current.filename().string() + ".tmp") {
        return Status::kInvalidConfig;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(staged, ec) || ec) return Status::kIoFailure;
    const std::filesystem::path backup =
        current.parent_path() / (current.filename().string() + ".old");
    std::filesystem::remove_all(backup, ec);
    if (ec) return Status::kIoFailure;
    const bool had_current = std::filesystem::exists(current, ec);
    if (ec) return Status::kIoFailure;
    if (had_current) {
        std::filesystem::rename(current, backup, ec);
        if (ec) return Status::kIoFailure;
    }
    std::filesystem::rename(staged, current, ec);
    if (ec) {
        if (had_current) {
            std::error_code restore_ec;
            std::filesystem::rename(backup, current, restore_ec);
        }
        return Status::kIoFailure;
    }
    if (had_current) {
        std::filesystem::remove_all(backup, ec);
        if (ec) return Status::kIoFailure;
    }
    return Status::kOk;
}

}  // namespace mgt
