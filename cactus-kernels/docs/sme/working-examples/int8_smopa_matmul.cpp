// INT8 SMOPA SME2 matmul (32x32x32, s8*s8 -> s32), verified against a scalar int32 triple
// loop on Apple M4. [ran-correct 2026-06-09] max_abs_err = 0, all 1024 cells exact, PASS.
//
// EXACT WORKING COMPILE COMMAND (Apple clang 17, M4):
//   clang++ -O2 -std=c++20 -march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2 \
//       int8_smopa_matmul.cpp -o int8_smopa_matmul && ./int8_smopa_matmul
//
// Also runs with -march=armv8-a+sme and -mcpu=apple-m4.  NOT -march=armv9-a+sme2 (SIGILL at run).
//
// Asm emitted (verified): smopa za0.s, p1/m, p2/m, z1.b, z0.b  (4-way widening, .b -> .s).
//
// ===== THE TWO BUGS THIS EXAMPLE EXISTS TO DOCUMENT =====
//
// 1. SMOPA WIDENING CONTRACT (empirically discovered with single-hot probes, NOT from docs):
//      ZA32[i][j] += sum_{c=0..3} zA[4*i + c] * zB[4*j + c]
//    i.e. each 32-bit lane GROUP of the byte vector holds 4 contiguous K values for one
//    output index. Proven: a[ai]=1,b[bj]=1 lights ZA[ai/4][bj/4] ONLY when ai%4 == bj%4
//    (the inner-K index c must match); all-ones gives 4 (sum of 4 products). So you must
//    pack:  aPanel[4*i + c] = A[(row+i)*K + (4g+c)]   (i = output row, c = inner-K 0..3)
//           bPanel[4*j + c] = B[(4g+c)*N + (col+j)]   (j = output col, c = inner-K 0..3)
//
// 2. PREDICATE WIDTH (the bug that made the first version produce 100% wrong cells):
//    SMOPA operands are BYTES, so the governing predicates pn/pm passed to svmopa_za32_s8_m
//    MUST be BYTE predicates (svptrue_b8 / svwhilelt_b8). Passing a b32 predicate (as you
//    would for the f32 FMOPA tile or the readout) marks only 1 of every 4 K-lanes active and
//    silently ZEROES the other 3 inner-K products -> garbage. The 32-bit predicate is only
//    for the svst1_hor_za32 readout (which writes .s lanes).
//    Tail handling for M/N is done by ZERO-PADDING the packed panels, so the byte predicate
//    can stay all-true.
#include <arm_sme.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

__arm_new("za") __arm_locally_streaming
static void matmul_s8_sme(uint64_t M, uint64_t K, uint64_t N,
                          const int8_t* A, const int8_t* B, int32_t* C) {
    const uint64_t SVLb = svcntb();           // bytes per Z (64 on M4)
    const uint64_t TILE = SVLb / 4;           // 16 int32 cells per tile side
    const svbool_t pB = svptrue_b8();         // BYTE predicate for the s8 SMOPA operands
    int8_t aPanel[64], bPanel[64];            // == SVLb
    for (uint64_t row = 0; row < M; row += TILE) {
        for (uint64_t col = 0; col < N; col += TILE) {
            svbool_t pN = svwhilelt_b32_u64(col, N);   // b32: for the readout store only
            svzero_za();
            for (uint64_t g = 0; g * 4 < K; ++g) {
                const uint64_t k0 = g * 4;
                for (uint64_t i = 0; i < TILE; ++i)
                    for (uint64_t c = 0; c < 4; ++c)
                        aPanel[4 * i + c] = (row + i < M && k0 + c < K)
                                          ? A[(row + i) * K + k0 + c] : 0;
                for (uint64_t j = 0; j < TILE; ++j)
                    for (uint64_t c = 0; c < 4; ++c)
                        bPanel[4 * j + c] = (col + j < N && k0 + c < K)
                                          ? B[(k0 + c) * N + (col + j)] : 0;
                svint8_t zA = svld1_s8(pB, &aPanel[0]);
                svint8_t zB = svld1_s8(pB, &bPanel[0]);
                svmopa_za32_s8_m(0, pB, pB, zA, zB);   // BYTE predicates! ZA[i][j]+=sum_c zA[4i+c]*zB[4j+c]
            }
            for (uint64_t r = 0; r < TILE && row + r < M; ++r)
                svst1_hor_za32(0, (uint32_t)r, pN, &C[(row + r) * N + col]);
        }
    }
}

static void matmul_s8_scalar(uint64_t M, uint64_t K, uint64_t N,
                             const int8_t* A, const int8_t* B, int32_t* C) {
    for (uint64_t i = 0; i < M; ++i)
        for (uint64_t j = 0; j < N; ++j) {
            int32_t s = 0;
            for (uint64_t k = 0; k < K; ++k) s += (int32_t)A[i * K + k] * (int32_t)B[k * N + j];
            C[i * N + j] = s;
        }
}

int main() {
    const uint64_t M = 32, K = 32, N = 32;
    std::vector<int8_t> A(M * K), B(K * N);
    std::vector<int32_t> Csme(M * N, 0), Cref(M * N, 0);
    srand(4321);
    for (auto& x : A) x = (int8_t)((rand() % 255) - 127);
    for (auto& x : B) x = (int8_t)((rand() % 255) - 127);

    matmul_s8_scalar(M, K, N, A.data(), B.data(), Cref.data());
    matmul_s8_sme(M, K, N, A.data(), B.data(), Csme.data());

    int64_t maxabs = 0; int mism = 0;
    for (uint64_t i = 0; i < M * N; ++i) {
        int64_t d = std::llabs((int64_t)Csme[i] - (int64_t)Cref[i]);
        if (d) ++mism;
        if (d > maxabs) maxabs = d;
    }
    printf("INT8 SMOPA matmul %llux%llux%llu  max_abs_err = %lld  (mismatches=%d)\n",
           (unsigned long long)M, (unsigned long long)K, (unsigned long long)N,
           (long long)maxabs, mism);
    bool ok = (maxabs == 0);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
