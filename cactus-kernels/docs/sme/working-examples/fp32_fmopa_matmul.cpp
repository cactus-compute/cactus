// FP32 FMOPA SME2 matmul (32x32x32), verified against a scalar triple loop on Apple M4.
// [ran-correct 2026-06-09] max_abs_err = 0.000e+00, PASS.
//
// EXACT WORKING COMPILE COMMAND (Apple clang 17, M4):
//   clang++ -O2 -std=c++20 -march=armv8.2-a+fp16+simd+dotprod+i8mm+sme2 \
//       fp32_fmopa_matmul.cpp -o fp32_fmopa_matmul && ./fp32_fmopa_matmul
//
// Also runs with -march=armv8-a+sme  and  -mcpu=apple-m4.
// DO NOT use -march=armv9-a+sme2 : it COMPILES and emits identical SME asm but the binary
// SIGILLs at runtime (exit 132) on M4 — armv9-a implies the non-streaming SVE2 ISA that Apple
// Silicon does not implement. Confirmed by running probe p6 across all four -march strings.
//
// Asm emitted (verified): smstart sm / smstart za / fmopa za0.s, p0/m, p1/m, z0.s, z1.s /
//                         st1w {za0h.s[wN, 0]}, p, [ptr] / smstop za / smstop sm.
//
// KEY GOTCHAS proven here:
//  * __arm_streaming / __arm_inout("za") are TYPE attributes -> they go AFTER the param list.
//    __arm_new("za") / __arm_locally_streaming are DECL attributes -> they go BEFORE the fn.
//    Putting __arm_streaming in the prefix position errors: "cannot be applied to a declaration".
//  * For f32 FMOPA the operands are 32-bit, so the governing predicates pn/pm are b32
//    (svwhilelt_b32 / svptrue_b32). FMOPA: ZA[i][j] += zn[i] * zm[j].
#include <arm_sme.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// C[M][N] = A[M][K] * B[K][N], all row-major.
// FMOPA per K is a rank-1 outer product: ZA[i][j] += a_col[i] * b_row[j]
//   a_col[i] = A[i][k]  (M-indexed column of A),  b_row[j] = B[k][j]  (N-indexed row of B).
__arm_new("za") __arm_locally_streaming
static void matmul_f32_sme(uint64_t M, uint64_t K, uint64_t N,
                           const float* A, const float* B, float* C) {
    const uint64_t SVL = svcntw();            // FP32 lanes in streaming mode (16 on M4)
    float acol[64];                            // >= SVL; gathers a column of A for k
    for (uint64_t row = 0; row < M; row += SVL) {
        svbool_t pM = svwhilelt_b32_u64(row, M);
        for (uint64_t col = 0; col < N; col += SVL) {
            svbool_t pN = svwhilelt_b32_u64(col, N);
            svzero_za();
            for (uint64_t k = 0; k < K; ++k) {
                for (uint64_t r = 0; r < SVL && row + r < M; ++r)
                    acol[r] = A[(row + r) * K + k];
                svfloat32_t zA = svld1_f32(pM, &acol[0]);
                svfloat32_t zB = svld1_f32(pN, &B[k * N + col]);
                svmopa_za32_f32_m(0, pM, pN, zA, zB);   // FMOPA: ZA[i][j] += zA[i]*zB[j]
            }
            for (uint64_t r = 0; r < SVL && row + r < M; ++r)
                svst1_hor_za32(0, (uint32_t)r, pN, &C[(row + r) * N + col]);
        }
    }
}

static void matmul_f32_scalar(uint64_t M, uint64_t K, uint64_t N,
                              const float* A, const float* B, float* C) {
    for (uint64_t i = 0; i < M; ++i)
        for (uint64_t j = 0; j < N; ++j) {
            float s = 0.f;
            for (uint64_t k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

int main() {
    const uint64_t M = 32, K = 32, N = 32;
    std::vector<float> A(M * K), B(K * N), Csme(M * N, 0.f), Cref(M * N, 0.f);
    srand(1234);
    for (auto& x : A) x = (float)((rand() % 200) - 100) / 32.0f;
    for (auto& x : B) x = (float)((rand() % 200) - 100) / 32.0f;

    matmul_f32_scalar(M, K, N, A.data(), B.data(), Cref.data());
    matmul_f32_sme(M, K, N, A.data(), B.data(), Csme.data());

    double maxabs = 0.0;
    for (uint64_t i = 0; i < M * N; ++i)
        maxabs = std::fmax(maxabs, std::fabs((double)Csme[i] - (double)Cref[i]));
    printf("FP32 FMOPA matmul %llux%llux%llu  max_abs_err = %.3e\n",
           (unsigned long long)M, (unsigned long long)K, (unsigned long long)N, maxabs);
    bool ok = maxabs < 1e-3;
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
