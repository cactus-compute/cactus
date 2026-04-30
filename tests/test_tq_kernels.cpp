#include "cactus_kernels.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum : uint32_t {
    OP_GEMV = 1,
    OP_GEMM = 2,
};

struct FixtureHeader {
    char magic[4];
    uint32_t version;
    uint32_t bits;
    uint32_t op;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t group_size;
    uint32_t num_groups;
    uint32_t flags;
};

struct Fixture {
    FixtureHeader header{};
    std::vector<__fp16> codebook;
    std::vector<int8_t> left_signs;
    std::vector<int8_t> right_signs;
    std::vector<uint32_t> permutation;
    std::vector<__fp16> norms;
    std::vector<uint8_t> packed;
    std::vector<__fp16> A;
    std::vector<__fp16> expected;
};

std::vector<uint8_t> load_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

bool read_bytes(const std::vector<uint8_t>& bytes, size_t& offset, void* dst, size_t count) {
    if (offset + count > bytes.size()) return false;
    std::memcpy(dst, bytes.data() + offset, count);
    offset += count;
    return true;
}

template<typename T>
bool read_vector(const std::vector<uint8_t>& bytes, size_t& offset, std::vector<T>& dst, size_t count) {
    dst.resize(count);
    return read_bytes(bytes, offset, dst.data(), count * sizeof(T));
}

bool load_fixture(const std::string& path, Fixture& fixture) {
    const std::vector<uint8_t> bytes = load_bytes(path);
    if (bytes.empty()) {
        std::cerr << "failed to open " << path << ": " << std::strerror(errno) << "\n";
        return false;
    }

    size_t offset = 0;
    if (!read_bytes(bytes, offset, &fixture.header, sizeof(FixtureHeader))) return false;
    if (std::memcmp(fixture.header.magic, "TQFX", 4) != 0 || fixture.header.version != 3) {
        std::cerr << path << ": bad fixture header\n";
        return false;
    }
    if (fixture.header.bits != 2 && fixture.header.bits != 4) {
        std::cerr << path << ": unsupported bits " << fixture.header.bits << "\n";
        return false;
    }
    if (fixture.header.K != fixture.header.group_size * fixture.header.num_groups) {
        std::cerr << path << ": invalid K/group dimensions\n";
        return false;
    }

    const size_t codebook_count = size_t{1} << fixture.header.bits;
    const size_t norms_count = static_cast<size_t>(fixture.header.N) * fixture.header.num_groups;
    const size_t packed_group_bytes =
        cactus_tq_packed_group_bytes(fixture.header.bits, fixture.header.group_size);
    const size_t packed_count =
        static_cast<size_t>(fixture.header.N) * fixture.header.num_groups * packed_group_bytes;
    const size_t A_count = static_cast<size_t>(fixture.header.M) * fixture.header.K;
    const size_t C_count = static_cast<size_t>(fixture.header.M) * fixture.header.N;

    if (!read_vector(bytes, offset, fixture.codebook, codebook_count)) return false;
    if (!read_vector(bytes, offset, fixture.left_signs, fixture.header.group_size)) return false;
    if (!read_vector(bytes, offset, fixture.right_signs, fixture.header.group_size)) return false;
    if (!read_vector(bytes, offset, fixture.permutation, fixture.header.group_size)) return false;
    if (!read_vector(bytes, offset, fixture.norms, norms_count)) return false;
    if (!read_vector(bytes, offset, fixture.packed, packed_count)) return false;
    if (!read_vector(bytes, offset, fixture.A, A_count)) return false;
    if (!read_vector(bytes, offset, fixture.expected, C_count)) return false;

    if (offset != bytes.size()) {
        std::cerr << path << ": trailing fixture bytes\n";
        return false;
    }
    return true;
}

bool same_half_bits(__fp16 a, __fp16 b) {
    uint16_t aa = 0;
    uint16_t bb = 0;
    std::memcpy(&aa, &a, sizeof(aa));
    std::memcpy(&bb, &b, sizeof(bb));
    return aa == bb;
}

std::string find_fixture(const std::string& name) {
    std::vector<std::string> roots;
    if (const char* env = std::getenv("CACTUS_TEST_ASSETS")) {
        roots.emplace_back(std::string(env) + "/tq_kernels");
    }
    roots.emplace_back("tests/assets/tq_kernels");
    roots.emplace_back("../tests/assets/tq_kernels");
    roots.emplace_back("../assets/tq_kernels");
    roots.emplace_back("assets/tq_kernels");

    for (const std::string& root : roots) {
        const std::string path = root + "/" + name;
        std::ifstream in(path, std::ios::binary);
        if (in) return path;
    }
    return {};
}

bool run_fixture(const std::string& name) {
    const std::string path = find_fixture(name);
    if (path.empty()) {
        std::cerr << "could not find fixture " << name << "\n";
        return false;
    }

    Fixture fixture;
    if (!load_fixture(path, fixture)) return false;

    CactusTQMatrix W{
        fixture.header.bits,
        fixture.header.K,
        fixture.header.N,
        fixture.header.group_size,
        fixture.header.num_groups,
        fixture.header.flags,
        fixture.codebook.data(),
        fixture.norms.data(),
        fixture.packed.data(),
        fixture.left_signs.data(),
        fixture.right_signs.data(),
        fixture.permutation.data(),
    };

    std::vector<__fp16> actual(fixture.expected.size(), static_cast<__fp16>(0));
    if (fixture.header.bits == 4 && fixture.header.op == OP_GEMV) {
        cactus_tq4_gemv(&W, fixture.A.data(), actual.data());
    } else if (fixture.header.bits == 4 && fixture.header.op == OP_GEMM) {
        cactus_tq4_gemm(&W, fixture.A.data(), fixture.header.M, actual.data());
    } else if (fixture.header.bits == 2 && fixture.header.op == OP_GEMV) {
        cactus_tq2_gemv(&W, fixture.A.data(), actual.data());
    } else if (fixture.header.bits == 2 && fixture.header.op == OP_GEMM) {
        cactus_tq2_gemm(&W, fixture.A.data(), fixture.header.M, actual.data());
    } else {
        std::cerr << path << ": unsupported op " << fixture.header.op << "\n";
        return false;
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!same_half_bits(actual[i], fixture.expected[i])) {
            std::cerr << path << ": output[" << i << "] expected "
                      << static_cast<float>(fixture.expected[i]) << " got "
                      << static_cast<float>(actual[i]) << "\n";
            return false;
        }
    }

    std::cout << "PASS " << path << "\n";
    return true;
}

}  // namespace

int main() {
    const char* fixtures[] = {
        "tq4_gemv.bin",
        "tq4_gemm.bin",
        "tq2_gemv.bin",
        "tq2_gemm.bin",
    };

    bool ok = true;
    for (const char* fixture : fixtures) {
        ok = run_fixture(fixture) && ok;
    }
    return ok ? 0 : 1;
}
