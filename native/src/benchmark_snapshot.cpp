#include "mgt/benchmark_snapshot.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <string>

namespace mgt {
namespace {

constexpr std::uint64_t kGolden = UINT64_C(0x9e3779b97f4a7c15);
constexpr std::uint64_t kStateDomain = UINT64_C(0x243f6a8885a308d3);
constexpr std::uint64_t kLabelDomain = UINT64_C(0x13198a2e03707344);
constexpr std::uint64_t kWeightDomain = UINT64_C(0xa4093822299f31d0);
constexpr std::uint64_t kWeightMDomain = UINT64_C(0x082efa98ec4e6c89);
constexpr std::uint64_t kWeightVDomain = UINT64_C(0x452821e638d01377);
constexpr std::uint64_t kAffineDomain = UINT64_C(0xbe5466cf34e90c6c);
constexpr std::uint64_t kAffineMDomain = UINT64_C(0xc0ac29b7c97c50dd);
constexpr std::uint64_t kAffineVDomain = UINT64_C(0x3f84d5b5b5470917);
constexpr std::uint64_t kRunningDomain = UINT64_C(0x9216d5d98979fb1b);

std::uint64_t Key(
    std::uint64_t seed,
    std::uint64_t domain,
    std::uint64_t index) {
    return seed ^ (domain + kGolden * index);
}

void FillSigned(
    std::vector<float>* values,
    std::uint64_t seed,
    std::uint64_t domain,
    float scale) {
    for (std::uint64_t i = 0; i < values->size(); ++i) {
        (*values)[i] = scale * SignedUnit(Key(seed, domain, i));
    }
}

}  // namespace

std::uint64_t SplitMix64(std::uint64_t value) {
    value += kGolden;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
}

float SignedUnit(std::uint64_t value) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>(SplitMix64(value) >> 40U);
    return static_cast<float>(bits) * (1.0f / 8388608.0f) - 1.0f;
}

Status FillBenchmarkSnapshot(
    std::uint64_t seed,
    std::uint64_t global_sample_offset,
    std::uint32_t active_rows,
    const BenchmarkSnapshotShape& shape,
    BenchmarkMutableState* mutable_state,
    std::vector<TrainStateStorage>* states,
    std::vector<float>* labels) {
    if (mutable_state == nullptr || states == nullptr || labels == nullptr ||
        active_rows == 0 || shape.state_len == 0 ||
        shape.state_len > kStateLen || shape.state_value_pad == 0 ||
        shape.state_value_pad > 256 || shape.output_dim == 0 ||
        shape.linear_parameter_count == 0 ||
        shape.batch_norm_feature_count == 0) {
        return Status::kInvalidConfig;
    }
    if (global_sample_offset >
        std::numeric_limits<std::uint64_t>::max() - (active_rows - 1ULL)) {
        return Status::kCapacityExceeded;
    }
    if (shape.batch_norm_feature_count >
            std::numeric_limits<std::size_t>::max() / 2ULL ||
        static_cast<std::uint64_t>(active_rows) >
            std::numeric_limits<std::size_t>::max() / shape.output_dim) {
        return Status::kCapacityExceeded;
    }

    const auto linear_count =
        static_cast<std::size_t>(shape.linear_parameter_count);
    const auto bn_features =
        static_cast<std::size_t>(shape.batch_norm_feature_count);
    const auto affine_count = 2ULL * bn_features;
    mutable_state->weights.resize(linear_count);
    mutable_state->weight_grad.assign(linear_count, 0.0f);
    mutable_state->weight_m.resize(linear_count);
    mutable_state->weight_v.resize(linear_count);
    mutable_state->affine.resize(affine_count);
    mutable_state->affine_grad.assign(affine_count, 0.0f);
    mutable_state->affine_m.resize(affine_count);
    mutable_state->affine_v.resize(affine_count);
    mutable_state->running.resize(affine_count);

    FillSigned(&mutable_state->weights, seed, kWeightDomain, 0.02f);
    FillSigned(&mutable_state->weight_m, seed, kWeightMDomain, 0.001f);
    FillSigned(&mutable_state->affine_m, seed, kAffineMDomain, 0.001f);
    for (std::size_t i = 0; i < linear_count; ++i) {
        mutable_state->weight_v[i] =
            0.0005f + 0.0005f *
                std::fabs(SignedUnit(Key(seed, kWeightVDomain, i)));
    }
    for (std::size_t i = 0; i < affine_count; ++i) {
        mutable_state->affine_v[i] =
            0.0005f + 0.0005f *
                std::fabs(SignedUnit(Key(seed, kAffineVDomain, i)));
        const float affine_noise =
            0.02f * SignedUnit(Key(seed, kAffineDomain, i));
        const float running_noise =
            0.05f * SignedUnit(Key(seed, kRunningDomain, i));
        mutable_state->affine[i] =
            i < bn_features ? 1.0f + affine_noise : affine_noise;
        mutable_state->running[i] =
            i < bn_features ? running_noise : 1.0f + running_noise;
    }

    states->assign(active_rows, TrainStateStorage{});
    labels->resize(static_cast<std::size_t>(active_rows) * shape.output_dim);
    for (std::uint32_t row = 0; row < active_rows; ++row) {
        const std::uint64_t global_row = global_sample_offset + row;
        for (std::uint32_t position = 0; position < shape.state_len; ++position) {
            const std::uint64_t index =
                global_row * kStateStorageLen + position;
            (*states)[row].v[position] = static_cast<StateValue>(
                SplitMix64(Key(seed, kStateDomain, index)) %
                shape.state_value_pad);
        }
        for (std::uint32_t output = 0; output < shape.output_dim; ++output) {
            const std::uint64_t index =
                global_row * shape.output_dim + output;
            (*labels)[static_cast<std::size_t>(row) * shape.output_dim + output] =
                SignedUnit(Key(seed, kLabelDomain, index));
        }
    }
    return Status::kOk;
}

}  // namespace mgt

namespace {

class Sha256 {
public:
    void Update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0) {
            const std::size_t take = std::min(size, block_.size() - block_size_);
            std::copy_n(data, take, block_.data() + block_size_);
            block_size_ += take;
            data += take;
            size -= take;
            if (block_size_ == block_.size()) {
                Transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<std::uint8_t, 32> Final() {
        const std::uint64_t total_bits = total_bytes_ * 8ULL;
        const std::uint8_t marker = 0x80;
        Update(&marker, 1);
        const std::uint8_t zero = 0;
        while (block_size_ != 56) Update(&zero, 1);
        std::array<std::uint8_t, 8> length{};
        for (unsigned i = 0; i < 8; ++i) {
            length[7U - i] = static_cast<std::uint8_t>(total_bits >> (8U * i));
        }
        Update(length.data(), length.size());
        std::array<std::uint8_t, 32> digest{};
        for (std::size_t i = 0; i < state_.size(); ++i) {
            digest[4 * i] = static_cast<std::uint8_t>(state_[i] >> 24U);
            digest[4 * i + 1] = static_cast<std::uint8_t>(state_[i] >> 16U);
            digest[4 * i + 2] = static_cast<std::uint8_t>(state_[i] >> 8U);
            digest[4 * i + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return digest;
    }

private:
    static std::uint32_t Rotate(std::uint32_t value, unsigned amount) {
        return (value >> amount) | (value << (32U - amount));
    }

    void Transform(const std::uint8_t* block) {
        static constexpr std::array<std::uint32_t, 64> k{
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::array<std::uint32_t, 64> w{};
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24U) |
                   (static_cast<std::uint32_t>(block[4 * i + 1]) << 16U) |
                   (static_cast<std::uint32_t>(block[4 * i + 2]) << 8U) |
                   static_cast<std::uint32_t>(block[4 * i + 3]);
        }
        for (unsigned i = 16; i < 64; ++i) {
            const std::uint32_t s0 = Rotate(w[i - 15], 7) ^ Rotate(w[i - 15], 18) ^ (w[i - 15] >> 3U);
            const std::uint32_t s1 = Rotate(w[i - 2], 17) ^ Rotate(w[i - 2], 19) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint32_t s1 = Rotate(e,6)^Rotate(e,11)^Rotate(e,25);
            const std::uint32_t ch = (e&f)^((~e)&g);
            const std::uint32_t t1 = h+s1+ch+k[i]+w[i];
            const std::uint32_t s0 = Rotate(a,2)^Rotate(a,13)^Rotate(a,22);
            const std::uint32_t maj = (a&b)^(a&c)^(b&c);
            const std::uint32_t t2 = s0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }

    std::array<std::uint32_t,8> state_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<std::uint8_t,64> block_{};
    std::size_t block_size_ = 0;
    std::uint64_t total_bytes_ = 0;
};

void HashU64(Sha256* hash, std::uint64_t value) {
    std::array<std::uint8_t,8> bytes{};
    for (unsigned i=0;i<8;++i) bytes[i]=static_cast<std::uint8_t>(value>>(8U*i));
    hash->Update(bytes.data(), bytes.size());
}

void HashFloats(Sha256* hash, const std::vector<float>& values) {
    HashU64(hash, values.size());
    for (float value : values) {
        const std::uint32_t bits=std::bit_cast<std::uint32_t>(value);
        std::array<std::uint8_t,4> bytes{
            static_cast<std::uint8_t>(bits),static_cast<std::uint8_t>(bits>>8U),
            static_cast<std::uint8_t>(bits>>16U),static_cast<std::uint8_t>(bits>>24U)};
        hash->Update(bytes.data(), bytes.size());
    }
}

}  // namespace

mgt::Status mgt::CanonicalBenchmarkSnapshotSha256(
    const BenchmarkMutableState& mutable_state,
    const std::vector<TrainStateStorage>& states,
    const std::vector<float>& labels,
    std::string* sha256_hex) {
    if (!sha256_hex) return Status::kInvalidConfig;
    Sha256 hash;
    static constexpr std::array<std::uint8_t,25> magic{
        'm','g','t','_','b','e','n','c','h','m','a','r','k','_','s','n','a','p','s','h','o','t','_','v','1'};
    hash.Update(magic.data(), magic.size());
    HashFloats(&hash, mutable_state.weights);
    HashFloats(&hash, mutable_state.weight_grad);
    HashFloats(&hash, mutable_state.weight_m);
    HashFloats(&hash, mutable_state.weight_v);
    HashFloats(&hash, mutable_state.affine);
    HashFloats(&hash, mutable_state.affine_grad);
    HashFloats(&hash, mutable_state.affine_m);
    HashFloats(&hash, mutable_state.affine_v);
    HashFloats(&hash, mutable_state.running);
    HashU64(&hash, states.size());
    for (const auto& state : states) hash.Update(state.v, sizeof(state.v));
    HashFloats(&hash, labels);
    const auto digest=hash.Final();
    static constexpr char hex[]="0123456789abcdef";
    sha256_hex->resize(64);
    for (std::size_t i=0;i<digest.size();++i) {
        (*sha256_hex)[2*i]=hex[digest[i]>>4U];
        (*sha256_hex)[2*i+1]=hex[digest[i]&15U];
    }
    return Status::kOk;
}
