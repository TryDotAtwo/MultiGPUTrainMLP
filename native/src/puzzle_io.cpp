#include "mgt/puzzle_io.hpp"
#include <array>
#include <cctype>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace mgt {
namespace {

std::vector<int> ExtractIntegers(const std::string& text) {
    std::vector<int> values;
    std::size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && !std::isdigit(static_cast<unsigned char>(text[i])) && text[i] != '-') {
            ++i;
        }
        if (i == text.size()) break;
        int sign = 1;
        if (text[i] == '-') {
            sign = -1;
            ++i;
        }
        int value = 0;
        bool has_digit = false;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            has_digit = true;
            value = value * 10 + (text[i] - '0');
            ++i;
        }
        if (has_digit) {
            values.push_back(sign * value);
        }
    }
    return values;
}

bool IsPermutation72(const std::array<bool, kStateLen>& seen) {
    for (bool value : seen) {
        if (!value) return false;
    }
    return true;
}

}  // namespace

Status LoadPuzzleDefinition(const std::filesystem::path& group_json,
                            const std::filesystem::path& target_bin,
                            PuzzleDefinition* out) {
    if (out == nullptr) return Status::kInvalidPuzzle;

    std::ifstream json_file(group_json, std::ios::binary);
    if (!json_file) return Status::kIoFailure;
    const std::string text((std::istreambuf_iterator<char>(json_file)),
                           std::istreambuf_iterator<char>());

    const std::vector<int> values = ExtractIntegers(text);
    const std::size_t required = 3 + static_cast<std::size_t>(kMoveCount) * kStateLen;
    if (values.size() < required) return Status::kInvalidPuzzle;
    if (values[0] != 888 || values[1] != static_cast<int>(kStateLen) ||
        values[2] != static_cast<int>(kMoveCount)) {
        return Status::kInvalidPuzzle;
    }

    std::size_t pos = 3;
    for (std::uint32_t move = 0; move < kMoveCount; ++move) {
        std::array<bool, kStateLen> seen{};
        for (std::uint32_t i = 0; i < kStateLen; ++i) {
            const int v = values[pos++];
            if (v < 0 || v >= static_cast<int>(kStateLen)) return Status::kInvalidPuzzle;
            if (seen[static_cast<std::size_t>(v)]) return Status::kInvalidPuzzle;
            seen[static_cast<std::size_t>(v)] = true;
            out->moves[move].v[i] = static_cast<StateValue>(v);
        }
        if (!IsPermutation72(seen)) return Status::kInvalidPuzzle;
        for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
            out->moves[move].v[i] = static_cast<StateValue>(i);
        }
    }

    std::ifstream target_file(target_bin, std::ios::binary);
    if (!target_file) return Status::kIoFailure;
    target_file.read(reinterpret_cast<char*>(out->target.v), kStateLen);
    if (target_file.gcount() != static_cast<std::streamsize>(kStateLen)) {
        return Status::kInvalidPuzzle;
    }
    char extra = 0;
    if (target_file.read(&extra, 1)) {
        return Status::kInvalidPuzzle;
    }
    for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i) {
        out->target.v[i] = 0;
    }

    return Status::kOk;
}

}  // namespace mgt