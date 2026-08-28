#include "mgt/puzzle_io.hpp"
#include <array>
#include <cctype>
#include <charconv>
#include <fstream>
#include <iterator>
#include <string>

namespace mgt {
namespace {

void SkipWhitespace(const std::string& text, std::size_t* position) {
    while (*position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[*position]))) ++*position;
}

bool FindUniqueMemberValue(
    const std::string& text, const char* key, std::size_t* value_position) {
    const std::string token = std::string("\"") + key + "\"";
    const auto first = text.find(token);
    if (first == std::string::npos || text.find(token, first + token.size()) != std::string::npos)
        return false;
    std::size_t position = first + token.size();
    SkipWhitespace(text, &position);
    if (position >= text.size() || text[position++] != ':') return false;
    SkipWhitespace(text, &position);
    *value_position = position;
    return true;
}

bool ParseUnsigned(const std::string& text, std::size_t* position, std::uint32_t* value) {
    const char* first = text.data() + *position;
    const char* last = text.data() + text.size();
    const auto result = std::from_chars(first, last, *value);
    if (result.ec != std::errc{} || result.ptr == first) return false;
    *position = static_cast<std::size_t>(result.ptr - text.data());
    return true;
}

bool Consume(const std::string& text, std::size_t* position, char expected) {
    SkipWhitespace(text, position);
    if (*position >= text.size() || text[*position] != expected) return false;
    ++*position;
    return true;
}

bool ParseMoveTable(const std::string& text, std::size_t position, PuzzleDefinition* out) {
    if (!Consume(text, &position, '[')) return false;
    for (std::uint32_t move = 0; move < kMoveCount; ++move) {
        if (!Consume(text, &position, '[')) return false;
        std::array<bool, kStateLen> seen{};
        for (std::uint32_t i = 0; i < kStateLen; ++i) {
            std::uint32_t value = 0;
            SkipWhitespace(text, &position);
            if (!ParseUnsigned(text, &position, &value) || value >= kStateLen || seen[value])
                return false;
            seen[value] = true;
            out->moves[move].v[i] = static_cast<StateValue>(value);
            if (i + 1U < kStateLen && !Consume(text, &position, ',')) return false;
        }
        if (!Consume(text, &position, ']')) return false;
        for (std::uint32_t i = kStateLen; i < kStateStorageLen; ++i)
            out->moves[move].v[i] = static_cast<StateValue>(i);
        if (move + 1U < kMoveCount && !Consume(text, &position, ',')) return false;
    }
    return Consume(text, &position, ']');
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

    std::size_t group_position = 0, state_position = 0, count_position = 0;
    std::uint32_t group_id = 0, state_len = 0, move_count = 0;
    if (!FindUniqueMemberValue(text, "group_id", &group_position) ||
        !FindUniqueMemberValue(text, "state_len", &state_position) ||
        !FindUniqueMemberValue(text, "move_count", &count_position) ||
        !ParseUnsigned(text, &group_position, &group_id) ||
        !ParseUnsigned(text, &state_position, &state_len) ||
        !ParseUnsigned(text, &count_position, &move_count) ||
        group_id != 888 || state_len != kStateLen || move_count != kMoveCount) {
        return Status::kInvalidPuzzle;
    }
    std::size_t moves_position = 0, actions_position = 0;
    const bool has_moves = FindUniqueMemberValue(text, "moves", &moves_position);
    const bool has_actions = FindUniqueMemberValue(text, "actions", &actions_position);
    if (has_moves == has_actions ||
        !ParseMoveTable(text, has_moves ? moves_position : actions_position, out))
        return Status::kInvalidPuzzle;

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

    if (!HasCanonicalInverseMovePairs(*out)) return Status::kInvalidPuzzle;

    return Status::kOk;
}

bool HasNonIdentityMove(const PuzzleDefinition& puzzle) {
    for (const auto& move : puzzle.moves) {
        for (std::uint32_t i = 0; i < kStateLen; ++i) {
            if (move.v[i] != static_cast<StateValue>(i)) return true;
        }
    }
    return false;
}

bool HasCanonicalInverseMovePairs(const PuzzleDefinition& puzzle) {
    if (kMoveCount % 2U != 0) return false;
    for (std::uint32_t first = 0; first < kMoveCount; first += 2U) {
        const auto& direct = puzzle.moves[first];
        const auto& inverse = puzzle.moves[first + 1U];
        for (std::uint32_t i = 0; i < kStateLen; ++i) {
            if (direct.v[inverse.v[i]] != i || inverse.v[direct.v[i]] != i)
                return false;
        }
    }
    return true;
}

}  // namespace mgt
